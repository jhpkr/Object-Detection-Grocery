import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np

# Load the YOLOv5 model
weights_path = "C:/Users/Joon Park/Desktop/School/UCHICAGO WORK/ML/final/yolov5/runs/train/exp11/weights/best.pt"

# Setup YOLOv5 with custom model weights
model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')  # 'source' set to 'local' means don't download anything but use local files


# Function to process a frame
def process_frame(frame):
    # Perform inference
    results = model(frame)

    # Render results
    results.render()

    # Convert the image to RGB (OpenCV uses BGR by default)
    processed_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB)
    
    detected_objects = results.pandas().xyxy[0]
    return processed_frame, detected_objects

# Streamlit application
st.title('YOLOv5 Object Detection on Camera Feed')

run = st.checkbox('Run')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Column layout
col1, col2 = st.columns(2)

if run:
    stframe = col1.empty()
    detected_text = col2.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        frame, detected_objects = process_frame(frame)

        # Display the frame in Streamlit
        stframe.image(frame, channels="RGB")

        # Display detected objects with prices
        object_info = ""
        for idx, row in detected_objects.iterrows():
            object_name = row['name']
            # You can update the price according to your data or logic
            price = 9.99  # Dummy price
            object_info += f"{object_name.capitalize()} - Price ${price}\n"
        
        detected_text.text(object_info)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
