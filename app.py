import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Vehicle classes we care about
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

st.set_page_config(layout="wide")
st.title("🚗 Smart Vehicle Detection System")

option = st.sidebar.selectbox(
    "Select Feature",
    ("Image Detection", "Video Detection", "Live Camera Detection")
)

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
                label = model.names[cls]
                if label in VEHICLE_CLASSES:
                    vehicle_count += 1

        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption="Detected Vehicles", use_column_width=True)
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

        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label in VEHICLE_CLASSES:
                        vehicle_count += 1

            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()

        st.success(f"Total Vehicles Detected in Video: {vehicle_count}")


# ===============================
# FEATURE 3 — LIVE CAMERA
# ===============================
elif option == "Live Camera Detection":

    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        vehicle_count = 0

        st.info("Press 'Stop' checkbox to stop camera")

        stop = st.checkbox("Stop")

        while cap.isOpened() and not stop:

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label in VEHICLE_CLASSES:
                        vehicle_count += 1

            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="BGR")

            time.sleep(0.03)

        cap.release()
        st.success(f"Total Vehicles Detected During Session: {vehicle_count}")