import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_best.pt')

st.title("Military Object Detection System")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    results = model(np.array(image))
    st.image(np.squeeze(results.render()), caption="Detection Output", use_column_width=True)
