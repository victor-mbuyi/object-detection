import streamlit as st
import os

# Définir la configuration de la page en premier
st.set_page_config(page_title="À propos",
                  layout='wide',
                  page_icon='./images/info.png')

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
        background-color: #87CEEB; /* Bleu ciel pour le panneau latéral */
        padding: 20px;
        border-right: 1px solid #D3D3D3;
    }
    .stButton>button {
        background: linear-gradient(90deg, #87CEEB, #FFFFFF); /* Dégradé bleu ciel à blanc */
        color: #1A1A1A;
        border: 1px solid #87CEEB;
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
    .stMarkdown {
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("À propos de l’application de détection et suivi d’objets avec YOLOv8")
st.markdown("""
## Aperçu du projet
Cette application utilise YOLOv8 pour la détection et le suivi avancés d’objets dans des images et vidéos, prenant en charge 20 classes d’objets avec un focus sur les véhicules d’urgence (voitures, bus) et les personnes. Le modèle a été fine-tuné sur un dataset personnalisé situé à `data_images/train` pour l’entraînement et `data_images/test` pour la validation. Les principales fonctionnalités incluent :

- **Détection d’objets** : Utilise un modèle YOLOv8 fine-tuné pour détecter 20 classes d’objets avec une haute précision.
- **Traitement d’images** : Téléchargez des images pour détecter des objets avec des boîtes englobantes et des scores de confiance.
- **Traitement de vidéos** : Téléchargez des vidéos pour une détection et un suivi frame par frame avec ByteTrack.
- **Suivi d’objets** : Implémente ByteTrack pour un suivi robuste avec des identifiants uniques à travers les frames.
- **Détection de sirène** : Analyse l’audio des vidéos pour détecter les sirènes, filtrant les faux positifs pour les véhicules d’urgence.
- **Vitesse et direction** : Estime la vitesse et la direction des véhicules suivis.
- **Journalisation** : Génère un fichier journal avec horodatages, coordonnées des boîtes englobantes, vitesse et direction.
- **Alertes** : Déclenche des alertes UI pour les véhicules d’urgence ou les personnes, renforcées par la détection de sirène.
- **Carte thermique** : Visualise les zones fréquentes de détection dans les vidéos.
- **Évaluation** : Mesure la précision de détection (Précision, Rappel, mAP) et les performances de suivi.

## Améliorations futures
- Intégration de Deep SORT pour un suivi basé sur l’apparence.
- Amélioration de l’estimation de vitesse/direction avec calibration de caméra.

Pour plus de détails, visitez les pages [Détection d’images](/YOLO_for_image/) ou [Détection de vidéos](/YOLO_for_video/).
""")