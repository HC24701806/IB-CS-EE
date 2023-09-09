from ultralytics import YOLO
from pathlib import Path
import os

# read NEPTUNE_API_TOKEN from .env text file
# and save it as environment variable
with open('.env') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=')
        os.environ[key] = value

# Neptune.ai logger is enabled in YOLO by default as long as the key is set in ENV

dataset = Path(__file__).parent / 'data' / 'lisav1' / 'lisa.yaml'
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(data=dataset, epochs=150, project='haolin.cong/TrafficSign-Yolo', name='lisav1')

