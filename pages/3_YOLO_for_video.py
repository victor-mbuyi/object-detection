import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolo_predictions import YOLO_Pred
from speed_direction import SpeedDirectionTracker
from audio_analysis1 import AudioAnalyzer
import time
import os
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader

# Définir la configuration de la page en premier
st.set_page_config(page_title="YOLOv8 Video Object Detection and Tracking",
                  layout='wide',
                  page_icon='./images/video.png')

# Appliquer un style CSS pour un design professionnel
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF; /* Fond principal en blanc */
        color: #1A1A1A; /* Texte noir foncé */
        font-family: 'Arial', sans-serif;
    }
    .css-1aumxhk { /* Sidebar */
        background-color: #0000FF; /* Bleu ciel pour le panneau latéral */
        padding: 20px;
        border-right: 1px solid #D3D3D3;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0000FF, #FFFFFF); /* Dégradé bleu ciel à blanc */
        color: #1A1A1A;
        border: 1px solid #0000FF;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #76B7D0, #E6F0FA);
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #87CEEB;
        border-radius: 5px;
        padding: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown {
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header('Détection et suivi d’objets dans les vidéos avec YOLOv8')
st.write('Sélectionnez un objet et téléchargez une vidéo pour obtenir les détections avec alarme')

try:
    with open('./models/data.yaml', mode='r') as f:
        data_yaml = yaml.load(f, Loader=SafeLoader)
    class_names = data_yaml['names']
except FileNotFoundError:
    st.error("Fichier data.yaml non trouvé dans le répertoire ./models/.")
    st.stop()

with st.spinner('Chargement du modèle YOLOv8...'):
    try:
        yolo = YOLO_Pred(model_path='./models/yolov8n.pt',
                         data_yaml='./models/data.yaml')
        speed_tracker = SpeedDirectionTracker(fps=30, pixel_to_meter=0.1)
        audio_analyzer = AudioAnalyzer()
    except FileNotFoundError:
        st.error("Fichier modèle yolov8n.pt non trouvé dans le répertoire ./models/.")
        st.stop()

def upload_video():
    video_file = st.file_uploader(label='Téléchargez une vidéo', type=['mp4', 'avi', 'mov'])
    if video_file is not None:
        size_mb = video_file.size / (1024 ** 2)
        file_details = {
            "filename": video_file.name,
            "filetype": video_file.type,
            "filesize": "{:,.2f} MB".format(size_mb)
        }
        return {"file": video_file, "details": file_details}
    return None

def process_frame(frame, yolo, speed_tracker, audio_analyzer, log_file, heatmap, frame_shape, target_class):
    pred_frame, detections = yolo.predict(frame, tracker='bytetrack.yaml', target_class=target_class)
    speeds, directions = speed_tracker.update(detections)
    
    timestamp = time.time()
    target_detected = False
    
    for det in detections:
        class_name = det['class_name']
        if class_name == target_class:
            target_detected = True
        x1, y1, x2, y2 = det['box']
        track_id = det['track_id']
        speed = speeds.get(track_id, 0)
        direction = directions.get(track_id, 0)
        # Agrandir la taille du texte pour les informations de vitesse et direction
        text = f'ID: {track_id} {class_name} Vitesse: {speed:.2f} m/s Dir: {direction:.1f}°'
        cv2.putText(pred_frame, text, (x1, y1-30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{track_id},{class_name},{x1},{y1},{x2},{y2},{speed:.2f},{direction:.1f}\n")
        
        heatmap[y1:y2, x1:x2] += 1
    
    return pred_frame, target_detected, heatmap

def main():
    target_class = st.selectbox("Sélectionnez l’objet à suivre avec alarme", options=class_names, index=0)
    video = upload_video()
    
    if video:
        st.subheader('Détails de la vidéo')
        st.json(video['details'])
        
        temp_file = f"temp_{video['file'].name}"
        try:
            with open(temp_file, 'wb') as f:
                f.write(video['file'].read())
            
            cap = cv2.VideoCapture(temp_file)
            if not cap.isOpened():
                st.error("Impossible d’ouvrir le fichier vidéo.")
                return
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
            
            log_file = 'detection_log.csv'
            with open(log_file, 'w') as f:
                f.write('timestamp,track_id,class_name,x1,y1,x2,y2,speed,direction\n')
            
            stframe = st.empty()
            alarm_placeholder = st.empty()
            detection_message_placeholder = st.empty()
            summary_placeholder = st.empty()
            
            alarm_audio_path = './audio/alarm.wav'
            audio_enabled = os.path.exists(alarm_audio_path)
            if not audio_enabled:
                st.warning("Fichier audio d’alarme non trouvé à './audio/alarm.wav'. Les alertes audio sont désactivées.")
            
            siren_detected = False
            if target_class in ['car', 'bus']:
                audio_file = audio_analyzer.extract_audio(temp_file)
                if audio_file:
                    siren_detected = audio_analyzer.detect_siren(audio_file)
                    audio_analyzer.cleanup(audio_file)
            
            frame_count = 0
            alert_triggered = False  # Track if alert has been triggered for this video
            target_ever_detected = False  # Track if target was ever detected
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred_frame, target_detected, heatmap = process_frame(
                    frame, yolo, speed_tracker, audio_analyzer, log_file, heatmap, (frame_height, frame_width), target_class
                )
                stframe.image(pred_frame)
                
                if target_detected:
                    if not target_ever_detected:
                        detection_message_placeholder.success(f"Objet {target_class} détecté dans la vidéo !")
                        target_ever_detected = True
                    if (target_class not in ['sheep', 'aeroplane'] or siren_detected) and not alert_triggered:
                        if audio_enabled:
                            alarm_placeholder.audio(alarm_audio_path, autoplay=True)
                        st.warning(f"ALERTE : {target_class} détecté ! Alarme déclenchée.")
                        alert_triggered = True
                else:
                    detection_message_placeholder.info(f"verification de {target_class}  dans cette frame.")
                
                frame_count += 1
            
            cap.release()
            os.remove(temp_file)
            
            # Résumé final
            if target_ever_detected:
                summary_placeholder.success(f"Résumé : L’objet {target_class} a été détecté dans la vidéo.")
            else:
                summary_placeholder.error(f"Résumé : L’objet {target_class} n’a pas été détecté dans la vidéo.")
            
            plt.figure(figsize=(10, 6))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Fréquence de détection')
            plt.title('Carte thermique des objets détectés')
            plt.savefig('heatmap.png')
            st.subheader('Carte thermique des objets détectés')
            st.image('heatmap.png')
            
            with open(log_file, 'r') as f:
                st.download_button('Télécharger le journal de détection', f, file_name='detection_log.csv')
        except Exception as e:
            st.error(f"Erreur lors du traitement de la vidéo : {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()