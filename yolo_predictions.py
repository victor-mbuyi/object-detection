#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
from ultralytics import YOLO
import os

class YOLO_Pred():
    def __init__(self, model_path, data_yaml):
        # Load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # Check for fine-tuned model, fall back to pre-trained model if not found
        fine_tuned_model_path = './models/yolov8n_finetuned.pt'
        self.model_path = fine_tuned_model_path if os.path.exists(fine_tuned_model_path) else model_path
        
        # Load YOLOv8 model
        self.model = YOLO(self.model_path)
        # Focus on emergency-related classes, but model supports all 20 classes
        self.focus_classes = ['person', 'car', 'bus']
    
    def predict(self, image, tracker='bytetrack.yaml', conf=0.4, iou=0.45, target_class=None):
        # Run YOLOv8 detection (no tracking) if tracker is None, otherwise use tracking
        if tracker is None:
            results = self.model.predict(image, conf=conf, iou=iou)
        else:
            results = self.model.track(image, conf=conf, iou=iou, tracker=tracker, persist=True)
        
        detections = []
        annotated_frame = image.copy()
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                class_id = int(box.cls.item())
                
                # Skip if class_id is out of range for self.labels
                if class_id >= len(self.labels):
                    continue
                
                class_name = self.labels[class_id]
                
                # Draw bounding box (blue for target_class, green otherwise)
                track_id = int(box.id.item()) if hasattr(box, 'id') and box.id is not None else -1
                color = (255, 0, 0) if class_name == target_class else (0, 255, 0)  # Blue for target, green for others
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f'ID: {track_id} {class_name}: {int(conf*100)}%' if track_id != -1 else f'{class_name}: {int(conf*100)}%'
                # Agrandir la taille de la police et l'épaisseur
                cv2.putText(annotated_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                    'track_id': track_id
                })
        
        return annotated_frame, detections