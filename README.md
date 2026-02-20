🚗 Vehicle Detection & Counting System using YOLOv8
📌 Project Overview

This project is an AI-based Vehicle Detection and Counting System built using Python, YOLOv8, OpenCV, and Streamlit.

The system can:

✅ Detect vehicles in an uploaded image

✅ Detect and count vehicles in an uploaded video

✅ Detect vehicles in real-time using a webcam

This project demonstrates practical applications of Computer Vision and Deep Learning in traffic monitoring and smart transportation systems.

🎯 Features
🔹 Feature 1 – Image Vehicle Detection

Upload an image

Detect vehicles (car, bus, truck, motorcycle)

Display bounding boxes

Show total vehicle count

🔹 Feature 2 – Video Vehicle Detection

Upload a road traffic video

Detect vehicles frame-by-frame

Count vehicles in the video

🔹 Feature 3 – Live Camera Detection

Use webcam

Real-time vehicle detection

Live counting display

🧠 Technologies Used

Python

YOLOv8 (Ultralytics)

OpenCV

Streamlit

NumPy

Pillow

🏗️ Project Structure
vehicle-detection-system/
│
├── app.py
├── requirements.txt
├── README.md
└── yolov8n.pt (auto-downloaded)
⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/Butsee155/vehicle-ai-system.git
cd vehicle-ai-system
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run app.py
📊 Dataset Used

The model uses the pre-trained YOLOv8 model trained on the COCO dataset.

Vehicle classes detected:

Car

Bus

Truck

Motorcycle

Optional datasets for further training:

https://cocodataset.org/

https://www.kaggle.com/datasets

🚀 Future Improvements

Vehicle brand classification

License plate recognition (OCR)

Speed estimation

Traffic density analysis

Deployment with cloud GPU

Model fine-tuning with custom dataset

🎓 Academic Value

This project demonstrates:

Object Detection

Deep Learning Model Integration

Real-Time Computer Vision

AI Deployment with Streamlit

End-to-End AI Application Development

👨‍💻 Author

Nisitha Nethsilu
BSc (Hons) in Data Science
GitHub: https://github.com/Butsee155

⭐ Why This Project Matters

This system can be applied in:

Smart Cities

Traffic Monitoring

Toll Systems

Parking Management

Surveillance Systems
