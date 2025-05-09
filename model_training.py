import torch
from yolov5 import train  # Assuming YOLOv5 repo is installed as a module

def train_model():
    train.run(
        data='data.yaml',
        imgsz=640,
        batch=16,
        epochs=50,
        weights='yolov5s.pt',
        name='military_model'
    )

if __name__ == "__main__":
    train_model()
