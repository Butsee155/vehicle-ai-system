import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
from sort_tracker import Sort
import easyocr
import torch
import torchvision.transforms as transforms
from torchvision import models

PIXELS_TO_METERS = 0.05  # Adjust if needed

# Load YOLO model
model = YOLO("yolov8n.pt")

# ===============================
# PHASE 4 MODELS
# ===============================

# OCR Reader
ocr_reader = easyocr.Reader(['en'])

# Simple Brand Classifier (Demo)
brand_model = models.mobilenet_v2(pretrained=True)
brand_model.eval()

brand_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# OCR Reader
ocr_reader = easyocr.Reader(['en'])

# Simple Brand Classifier (Pretrained MobileNet)
brand_model = models.mobilenet_v2(pretrained=True)
brand_model.eval()

brand_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Correct vehicle classes in COCO dataset
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

st.set_page_config(layout="wide")
st.title("🚗 Smart Vehicle Detection System")

option = st.sidebar.selectbox(
    "Select Feature",
    ("Image Detection", "Video Detection", "Live Camera Detection")
)

CONF_THRESHOLD = 0.4

# ===============================
# FEATURE 1 — IMAGE DETECTION
# ===============================
if option == "Image Detection":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        results = model(image_np)

        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if label in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
                    vehicle_count += 1

        annotated_frame = results[0].plot()

        st.image(annotated_frame, use_column_width=True)
        st.success(f"Total Vehicles Detected: {vehicle_count}")


# ===============================
# FEATURE 2 — VIDEO DETECTION
# ===============================
elif option == "Video Detection":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        vehicle_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
                        vehicle_count += 1

            annotated_frame = results[0].plot()
            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
        st.success(f"Total Vehicles Detected in Video: {vehicle_count}")


# ===============================
# FEATURE 3 — LIVE TRACKING + SPEED + RECOGNITION
# ===============================
elif option == "Live Camera Detection":

    st.subheader("🚦 Smart Traffic Monitoring (Full AI System)")

    tracker = Sort()

    CONF_THRESHOLD = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05)

    selected_camera = st.selectbox("Select Camera", [0,1,2])

    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)

        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        counted_ids = set()
        direction_memory = {}
        position_memory = {}
        time_memory = {}

        total_crossed = 0
        line_position = 350

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            detections = []
            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        detections.append([x1,y1,x2,y2])

            tracked_objects = tracker.update(detections)

            cv2.line(frame, (0,line_position),
                     (frame.shape[1],line_position),
                     (0,255,255), 2)

            for obj in tracked_objects:

                x1,y1,x2,y2,obj_id = obj
                x1,y1,x2,y2,obj_id = int(x1),int(y1),int(x2),int(y2),int(obj_id)

                cx = (x1+x2)//2
                cy = (y1+y2)//2

                current_time = time.time()
                speed_kmh = 0

                # SPEED CALCULATION
                if obj_id in position_memory:

                    prev_cx, prev_cy = position_memory[obj_id]
                    prev_time = time_memory[obj_id]

                    pixel_distance = np.sqrt(
                        (cx-prev_cx)**2 + (cy-prev_cy)**2
                    )

                    time_diff = current_time - prev_time

                    if time_diff > 0:
                        meters = pixel_distance * PIXELS_TO_METERS
                        speed_mps = meters / time_diff
                        speed_kmh = speed_mps * 3.6

                position_memory[obj_id] = (cx,cy)
                time_memory[obj_id] = current_time

                # Draw bounding box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                            f"ID {obj_id} | {int(speed_kmh)} km/h",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,0),2)

                # ===============================
                # PHASE 4 — BRAND + OCR
                # ===============================

                vehicle_crop = frame[y1:y2, x1:x2]

                brand_name = "Unknown"
                plate_text = "Not Detected"

                # BRAND CLASSIFICATION (Demo)
                try:
                    input_tensor = brand_transform(vehicle_crop).unsqueeze(0)
                    with torch.no_grad():
                        output = brand_model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        brand_name = f"BrandID {predicted.item()}"
                except:
                    pass

                # LICENSE PLATE OCR
                try:
                    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
                    results_ocr = ocr_reader.readtext(gray)
                    if len(results_ocr) > 0:
                        plate_text = results_ocr[0][1]
                except:
                    pass

                overspeed_alert = ""
                if speed_kmh > 80:
                    overspeed_alert = "🚨 OVER SPEED!"

                cv2.putText(frame,
                            f"{brand_name} | {plate_text}",
                            (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,255),2)

                if overspeed_alert:
                    cv2.putText(frame,
                                overspeed_alert,
                                (x1,y2+40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),2)

                # Save Logs
                with open("traffic_logs.csv","a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{obj_id},{int(speed_kmh)},{brand_name},{plate_text}\n")

                # Line Crossing
                if obj_id in direction_memory:
                    prev_y = direction_memory[obj_id]

                    if prev_y < line_position and cy >= line_position:
                        if obj_id not in counted_ids:
                            total_crossed += 1
                            counted_ids.add(obj_id)

                direction_memory[obj_id] = cy

            density = "🟢 Low"
            if total_crossed > 5:
                density = "🟡 Medium"
            if total_crossed > 10:
                density = "🔴 Heavy"

            frame_placeholder.image(frame, channels="BGR")

            stats_placeholder.markdown(f"""
            ### 🚘 Vehicles Crossed Line: {total_crossed}
            📈 Traffic Density: {density}
            """)

        cap.release()
