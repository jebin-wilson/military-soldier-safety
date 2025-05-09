import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_best.pt')

def detect_video(source=0):
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        cv2.imshow('Detection', np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_video()
