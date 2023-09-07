# imports
import logging
import pathlib
import sys
import warnings
from typing import List, Dict
import napari

import numpy as np
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn_tutorial.datasets import CreateMLDataset
from pytorch_faster_rcnn_tutorial.transformations import (
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import get_filenames_of_path
from pytorch_faster_rcnn_tutorial.viewers.object_detection_viewer import ObjectDetectionViewer

warnings.filterwarnings("ignore")

logger: logging.Logger = logging.getLogger(__name__)

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# root directory
# root directory (project directory)
ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.absolute()
train_path: pathlib.Path = (
    ROOT_PATH / "data" / "lisav1" / "train" / "images"
)

# mapping
mapping: Dict[str, int] = {
    'signalAhead': 1,
    'speedLimit25': 2,
    'pedestrianCrossing': 3,
    'school': 4,
    'noLeftTurn': 5,
    'slow': 6,
    'yield': 7,
    'laneEnds': 8,
    'turnRight': 9,
    'speedLimit35': 10,
    'schoolSpeedLimit25': 11,
    'rightLaneMustTurn': 12,
    'addedLane': 13,
    'keepRight': 14,
    'rampSpeedAdvisory50': 15,
    'stop': 16,
    'speedLimit40': 17,
    'merge': 18,
    'stopAhead': 19,
    'speedLimitUrdbl': 20,
    'speedLimit45': 21,
    'speedLimit55': 22,
    'noRightTurn': 23,
    'speedLimit30': 24,
    'yieldAhead': 25,
    'speedLimit50': 26,
    'roundabout': 27,
    'truckSpeedLimit55': 28,
    'curveRight': 29,
    'speedLimit65': 30,
    'dip': 31,
    'turnLeft': 32,
    'rampSpeedAdvisory20': 33,
    'curveLeft': 34,
    'thruTrafficMergeLeft': 35,
    'rampSpeedAdvisory45': 36,
    'zoneAhead25': 37,
    'doNotEnter': 38,
    'rampSpeedAdvisory40': 39,
    'speedLimit15': 40,
    'zoneAhead45': 41,
    'thruMergeRight': 42,
    'doNotPass': 43,
    'rampSpeedAdvisoryUrdbl': 44,
    'rampSpeedAdvisory35': 45,
    'thruMergeLeft': 46,
    'intersection': 47,
    }

# transformations
transforms: ComposeDouble = ComposeDouble(
    [
        Clip(),
        # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
        # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
        # AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01),
    ]
)

# dataset building
dataset: CreateMLDataset = CreateMLDataset(
    inputPath=train_path,
    transform=transforms,
    convert_to_format=None,
    mapping=mapping,
)

# transforms
transform: GeneralizedRCNNTransform = GeneralizedRCNNTransform(
    min_size=1024,
    max_size=1024,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

# visualize dataset
color_mapping: Dict[int, str] = dict(zip(mapping.values(), ['red'] * len(mapping)))

object_detection_viewer_rcnn: ObjectDetectionViewer = ObjectDetectionViewer(
    dataset=dataset, color_mapping=color_mapping, rcnn_transform=transform
)

napari.run()
