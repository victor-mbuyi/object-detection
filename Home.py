import streamlit as st 

# Définir la configuration de la page en premier
st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

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

st.title("Application de détection et suivi d’objets avec YOLOv8")
st.caption('Cette application web démontre la détection et le suivi d’objets avec YOLOv8')

# Content
st.markdown("""
### Cette application détecte et suit des objets dans des images et vidéos
- Utilise YOLOv8 pour une détection et un suivi améliorés de 20 objets, avec un focus sur les véhicules d’urgence (voitures, bus) et les personnes
- Détection basée sur des images : [Cliquez ici pour l’application d’images](/YOLO_for_image/)
- Détection et suivi basés sur des vidéos : [Cliquez ici pour l’application de vidéos](/YOLO_for_video/)
- Fonctionnalités :
  - Affichage en temps réel des boîtes englobantes avec des identifiants de suivi uniques grâce à ByteTrack
  - Détection de sirène pour confirmer les véhicules d’urgence
  - Estimation de la vitesse et de la direction des véhicules suivis
  - Génération d’un fichier journal avec horodatages, coordonnées des boîtes englobantes, vitesse et direction
  - Alertes lorsque des véhicules d’urgence ou des personnes sont détectés
  - Visualisation par carte thermique des objets détectés dans les vidéos

Voici les objets que notre modèle détecte :
1. Personne
2. Voiture
3. Chaise
4. Bouteille
5. Plante en pot
6. Oiseau
7. Chien
8. Canapé
9. Vélo
10. Cheval
11. Bateau
12. Moto
13. Chat
14. Moniteur TV
15. Vache
16. Mouton
17. Avion
18. Train
19. Table à manger
20. Bus
""")