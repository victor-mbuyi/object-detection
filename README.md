YOLOv8 Object Detection and Tracking Application
-------------------------------------------------
Welcome to the YOLOv8 Object Detection and Tracking Application! This project was designed by students from the African Institute for Mathematical Sciences (AIMS) Senegal as part of the Computer Vision course, under the guidance and contribution of Victor Mbuyi, Rihana Bankole, and Aly Tidiga. It is a web-based tool built with Streamlit and powered by the YOLOv8 algorithm for object detection and ByteTrack for multi-object tracking. It focuses on detecting and tracking 20 object classes, with an emphasis on emergency vehicles (cars, buses) and persons, and includes advanced features like speed/direction estimation, siren detection, and heatmap visualization.
Overview
This application allows users to upload images or videos to detect and track objects in real-time. It was developed as a proof-of-concept for intelligent surveillance and emergency response systems, leveraging a custom-trained YOLOv8 model on a dataset of 2,000 images and 50 videos collected from the internet and annotated using LabelImg.
Key Features

Object Detection: Detects 20 classes (e.g., person, car, bus, chair, etc.) using a fine-tuned YOLOv8 model.
Object Tracking: Implements ByteTrack for robust tracking across video frames with unique IDs.
Image Processing: Upload images to view detected objects with bounding boxes and confidence scores.
Video Analysis: Process videos with real-time tracking, speed/direction estimation, and heatmap generation.
Siren Detection: Analyzes video audio to confirm emergency vehicles with siren sounds.
Alerts: Triggers UI alerts for detected emergency vehicles or persons.
Logging: Generates a CSV log file with timestamps, bounding box coordinates, speed, and direction.
Visualization: Displays heatmaps to highlight frequent detection areas.



Install Dependencies:
---------------------
Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required packages:pip install -r requirements.txt




Install Additional Tools:
------------------------
OpenCV: For video processing (pip install opencv-python).
Ultralytics YOLOv8: For object detection (pip install ultralytics).
Streamlit: For the web interface (pip install streamlit).
Matplotlib: For heatmap visualization (pip install matplotlib).
ffmpeg: For audio extraction (install via your package manager, e.g., brew install ffmpeg on macOS).


Prepare the Dataset:
---------------------
Place your annotated images and videos in the data_images/train and data_images/test folders.
Update models/data.yaml with the correct paths and class names.


Fine-Tune the Model:

Run the fine-tuning script to generate yolov8n_finetuned.pt:python fine_tune_yolo.py





Usage
-----
Launch the Application:

Start the Streamlit app:streamlit run Home.py




Navigate the Interface:
-----------------------
Home: Overview of the application.
YOLO for Image: Upload images for object detection.
YOLO for Video: Upload videos for detection, tracking, and analysis.
About: Project details and future improvements.


Interact with Features:
-----------------------
Select a target object from the dropdown menu.
Upload an image or video to see real-time results, including bounding boxes, speed/direction, and alerts.



             


Contact
-------
For questions or support, please open an issue on this repository or contact mbuyi.b.victor@aims-senegal.org

