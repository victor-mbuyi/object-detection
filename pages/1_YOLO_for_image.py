import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import yaml
from yaml.loader import SafeLoader
import os

# Définir la configuration de la page en premier
st.set_page_config(page_title="YOLOv8 Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

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

st.header('Détection d’objets dans les images avec YOLOv8')
st.write('Sélectionnez un objet et téléchargez une image pour obtenir les détections')

# Load class names from data.yaml
with open('./models/data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)
class_names = data_yaml['names']

with st.spinner('Chargement du modèle YOLOv8...'):
    yolo = YOLO_Pred(model_path='./models/yolov8n.pt',
                     data_yaml='./models/data.yaml')

def upload_image():
    image_file = st.file_uploader(label='Téléchargez une image')
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": "{:,.2f} MB".format(size_mb)
        }
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Type de fichier IMAGE VALIDE (png ou jpeg)')
            return {"file": image_file, "details": file_details}
        else:
            st.error('Type de fichier image NON VALIDE. Téléchargez uniquement des fichiers png, jpg ou jpeg.')
            return None

def main():
    # Add dropdown for selecting target object
    target_class = st.selectbox("Sélectionnez l’objet à détecter", options=class_names, index=0)
    
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        col1, col2 = st.columns(2)
        
        with col1:
            st.info('Aperçu de l’image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Détails du fichier')
            st.json(object['details'])
            button = st.button('Obtenir la détection avec YOLOv8')
            
            # Load alarm audio
            alarm_audio_path = './audio/alarm.wav'
            if not os.path.exists(alarm_audio_path):
                st.error("Fichier audio d’alarme non trouvé. Veuillez placer 'alarm.wav' dans le répertoire 'audio/'.")
                return
            
            detection_message_placeholder = st.empty()
            if button:
                with st.spinner('Détection des objets...'):
                    image_array = np.array(image_obj)
                    pred_img, detections = yolo.predict(image_array, tracker=None, target_class=target_class)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
                    
                    # Check for target object
                    target_detected = any(d['class_name'] == target_class for d in detections)
                    if target_detected:
                        st.warning(f"ALERTE : {target_class} détecté ! Alarme déclenchée.")
                        st.audio(alarm_audio_path, autoplay=True)
                        detection_message_placeholder.success(f"Objet {target_class} détecté dans l’image !")
                    else:
                        detection_message_placeholder.error(f"Objet {target_class} non détecté dans l’image.")
        
        if prediction:
            st.subheader("Image avec prédictions")
            st.caption("Détection d’objets avec le modèle YOLOv8")
            st.image(pred_img_obj)

if __name__ == "__main__":
    main()