from ultralytics import YOLO
from pathlib import Path
import os
from pydantic import BaseSettings, Field

class NeptuneSettings(BaseSettings):
    """
    Reads the variables from the environment.
    Errors will be raised if the required variables are not set.
    """

    api_key: str = Field(default=..., env="NEPTUNE")
    PROJECT: str = "haolin.cong/TrafficSign-Yolo"
    EXPERIMENT: str = "traffic"

    class Config:
        # this tells pydantic to read the variables from the .env file
        env_file = ".env"

neptune_settings = NeptuneSettings()
os.environ["NE"] = neptune_settings.api_key

# Neptune.ai logger is enabled in YOLO by default as long as the key is set in ENV

dataset = Path(__file__).parent / 'data' / 'lisav1' / 'lisa.yaml'
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(data=dataset, epochs=150, project=neptune_settings.PROJECT, name='lisav1')

