
import json
from pathlib import Path

class LisaSet():
    ROOT_PATH: Path = Path(__file__).parent.absolute() / "data" / "lisav1"
    TRAIN_PATH: Path = ROOT_PATH / "train"
    VAL_PATH: Path = ROOT_PATH / "val"
    TEST_PATH: Path = ROOT_PATH / "test"
    TRAIN_IMAGES_PATH: Path = TRAIN_PATH / "images"
    VAL_IMAGES_PATH: Path = VAL_PATH / "images"
    TEST_IMAGES_PATH: Path = TEST_PATH / "images"

    @staticmethod
    def get_labels():
        with open(LisaSet.ROOT_PATH / "labels.json") as f:
            return json.load(f)
