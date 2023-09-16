from ultralytics import YOLO
from pathlib import Path
import os

class NeptuneSettings():
    """
    Reads the variables from .env file.
    Errors will be raised if the required variables are not set.
    """
    PROJECT: str = "haolin.cong/TrafficSign-Yolo"
    def __init__(self):
        with open(Path(__file__).parent / '.env', 'r') as f:
            for line in f.readlines():
                key, value = line.strip().split('=', maxsplit=1)
                if key == 'NEPTUNE_API_TOKEN':
                    self.api_key = value

neptune_settings = NeptuneSettings()
os.environ["NEPTUNE_API_TOKEN"] = neptune_settings.api_key

# Neptune.ai logger is enabled in YOLO by default as long as the key is set in ENV

dataset = Path(__file__).parent / 'data' / 'lisav1' / 'lisa.yaml'
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(data=dataset, epochs=150, project=neptune_settings.PROJECT, name='lisav1')

