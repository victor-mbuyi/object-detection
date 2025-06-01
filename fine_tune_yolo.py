import yaml
from ultralytics import YOLO
from pathlib import Path
import os

def fine_tune_model(model_path='./models/yolov8n.pt', data_yaml='./models/data.yaml', epochs=50, imgsz=640, batch=8):
    """
    Fine-tune a YOLOv8 model on a custom dataset with data augmentation.
    """
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='yolov8n_finetuned',
        project='./models',
        device=0 if os.environ.get('CUDA_AVAILABLE') else 'cpu',
        augment=True,  # Activer les augmentations (flip, rotation, etc.)
        hsv_h=0.015,   # Variation de teinte
        hsv_s=0.7,     # Variation de saturation
        hsv_v=0.4,     # Variation de valeur
        degrees=15.0,  # Rotation
        translate=0.1, # Décalage
        scale=0.5      # Échelle
    )
    fine_tuned_model_path = './models/yolov8n_finetuned.pt'
    model.save(fine_tuned_model_path)
    return fine_tuned_model_path

def evaluate_model(model_path, data_yaml, split='val'):
    """
    Evaluate the model on the validation set and return metrics.
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split)
    precision = metrics.results_dict.get('metrics/precision(B)', 0)
    recall = metrics.results_dict.get('metrics/recall(B)', 0)
    mAP_50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    mAP_50_95 = metrics.results_dict.get('metrics/mAP50:95(B)', 0)
    return {
        'precision': precision,
        'recall': recall,
        'mAP_50': mAP_50,
        'mAP_50_95': mAP_50_95
    }

def main():
    model_path = './models/yolov8n.pt'
    data_yaml = './models/data.yaml'
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    if not os.path.exists(data_yaml):
        print(f"Erreur : Le fichier {data_yaml} n'existe pas.")
        return
    
    with open(data_yaml, mode='r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    train_path = os.path.join(project_root, data['train'].lstrip('./'))
    val_path = os.path.join(project_root, data['val'].lstrip('./'))
    
    train_images_path = os.path.join(train_path, 'images')
    train_labels_path = os.path.join(train_path, 'labels')
    if not os.path.exists(train_images_path):
        print(f"Erreur : Le dossier des images d'entraînement {train_images_path} n'existe pas.")
        return
    if not os.path.exists(train_labels_path):
        print(f"Erreur : Le dossier des annotations d'entraînement {train_labels_path} n'existe pas.")
        return
    
    val_images_path = os.path.join(val_path, 'images')
    val_labels_path = os.path.join(val_path, 'labels')
    if not os.path.exists(val_images_path):
        print(f"Erreur : Le dossier des images de validation {val_images_path} n'existe pas.")
        return
    if not os.path.exists(val_labels_path):
        print(f"Erreur : Le dossier des annotations de validation {val_labels_path} n'existe pas.")
        return
    
    train_images = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.png'))]
    train_labels = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]
    if not train_images:
        print(f"Erreur : Aucune image trouvée dans {train_images_path}.")
        return
    if not train_labels:
        print(f"Erreur : Aucune annotation trouvée dans {train_labels_path}.")
        return
    
    val_images = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.png'))]
    val_labels = [f for f in os.listdir(val_labels_path) if f.endswith('.txt')]
    if not val_images:
        print(f"Erreur : Aucune image trouvée dans {val_images_path}.")
        return
    if not val_labels:
        print(f"Erreur : Aucune annotation trouvée dans {val_labels_path}.")
        return
    
    print("Démarrage du fine-tuning...")
    fine_tuned_model_path = fine_tune_model(model_path, data_yaml, epochs=50, imgsz=640, batch=8)
    print(f"Modèle fine-tuné sauvegardé à : {fine_tuned_model_path}")
    
    print("Évaluation du modèle fine-tuné...")
    metrics = evaluate_model(fine_tuned_model_path, data_yaml, split='val')
    print("Métriques d'évaluation :")
    print(f"Précision : {metrics['precision']:.3f}")
    print(f"Rappel : {metrics['recall']:.3f}")
    print(f"mAP@0.5 : {metrics['mAP_50']:.3f}")
    print(f"mAP@0.5:0.95 : {metrics['mAP_50_95']:.3f}")

if __name__ == "__main__":
    main()