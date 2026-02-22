🚦 AI-Powered Smart Traffic Monitoring & Vehicle Recognition System
🔥 Project Overview

This project is a real-time AI-based Smart Traffic Monitoring System built using YOLOv8, Object Tracking (SORT), Speed Estimation, and OCR.

The system detects, tracks, analyzes, and recognizes vehicles in real-time using computer vision and deep learning techniques.

It simulates a real-world intelligent traffic surveillance system used in:

Smart Cities

Highway Monitoring

Toll Gates

Traffic Law Enforcement

Urban Mobility Analytics

🎯 Key Features
🚘 1. Vehicle Detection

YOLOv8-based real-time vehicle detection

Supports: Car, Truck, Bus, Motorcycle

Adjustable confidence threshold

🔄 2. Unique Vehicle Tracking

SORT tracking algorithm

Unique ID assignment per vehicle

No duplicate counting

📏 3. Line Crossing Counter

Counts vehicles only when crossing a virtual line

Eliminates repeated frame counting

🧭 4. Direction Detection

Detects vehicle movement direction

Tracks vertical crossing logic

🚀 5. Speed Estimation

Real-time speed calculation (km/h approximation)

Pixel-to-meter conversion logic

Overspeed alert system

🏷 6. Vehicle Recognition Layer

Demo brand classification pipeline (MobileNetV2)

License Plate OCR using EasyOCR

Text extraction from vehicle region

📊 7. Traffic Analytics

Traffic density classification (Low / Medium / Heavy)

Live performance metrics

CSV logging of:

Timestamp

Vehicle ID

Speed

Brand ID

Plate number

🛠 Technologies Used

Python

YOLOv8 (Ultralytics)

OpenCV

SORT Tracking Algorithm

PyTorch

MobileNetV2

EasyOCR

NumPy

Streamlit (UI Layer)

🧠 System Architecture

Detection → Tracking → Speed Estimation → Recognition → Analytics → Logging

This multi-model pipeline demonstrates integration of:

Deep Learning

Computer Vision

Object Tracking

OCR

Real-time Data Processing

📂 Project Capabilities

✔ Real-time vehicle detection
✔ Unique ID tracking
✔ Line crossing logic
✔ Speed estimation
✔ Over-speed alert
✔ License plate reading
✔ Traffic density analysis
✔ Log generation for analytics

🎓 Academic & Professional Value

This project demonstrates:

End-to-end AI system design

Real-time inference pipeline

Multi-model integration

Applied computer vision engineering

Smart city technology simulation

🚀 Future Improvements

Custom-trained vehicle brand classifier

YOLO-based license plate detector

Camera calibration for accurate speed measurement

Multi-camera dashboard

Cloud deployment (Docker + GPU server)

REST API version (Flask/FastAPI)

👨‍💻 Author

Nisitha Nethsilu
BSc (Hons) in Data Science
Aspiring AI Engineer | Computer Vision Enthusiast
