
from collections import OrderedDict
import json
import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_mobilenet_v3_large_fpn
from pytorch_faster_rcnn_tutorial.datasets import CreateMLDataset, ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.faster_RCNN import (
    FasterRCNNLightning,
    get_faster_rcnn_resnet,
    get_fasterrcnn_mobilenet_v3_large_fpn,
)
from pytorch_faster_rcnn_tutorial.transformations import ComposeDouble
from pytorch_faster_rcnn_tutorial.transformations import (
    AlbumentationWrapper,
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_double,
)

# root directory (project directory)
ROOT_PATH: Path = Path(__file__).parent.absolute()

# Change this to the path of the model checkpoint you want to evaluate
model_path: Path = Path(ROOT_PATH) / "src/pytorch_faster_rcnn_tutorial/models/fasterrcnn_mobilenet_v3_large_fpn.ckpt"
test_path: Path = ROOT_PATH / "src" / "pytorch_faster_rcnn_tutorial" / "data" / "lisav1" / "test" / "images"

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

# test transformations
transforms_test: ComposeDouble = ComposeDouble(
    [
        Clip(),
        FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(function=normalize_01),
    ]
)

# dataset test
dataset_test: ObjectDetectionDataSet = CreateMLDataset(
    inputPath=test_path,
    transform=transforms_test,
    convert_to_format=None,
    mapping=mapping,
)

# dataloader test
dataloader_test: DataLoader = DataLoader(
    dataset=dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_double,
)

# model
model: FasterRCNN = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=len(mapping) + 1)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# remove the model. prefix in saved checkpoint
state_dict = OrderedDict([(re.sub('^model\\.', '', k), v) for k, v in checkpoint['state_dict'].items()])
model.load_state_dict(state_dict)
model.eval()

count = 0
total_time = 0

# https://cocodataset.org/#format-results
coco_results = []

with torch.no_grad():
    for images, _, x_name, _ in dataloader_test:
        _b = time.perf_counter_ns()
        predictions = model(images)
        _e = time.perf_counter_ns()
        total_time += _e - _b
        count += 1
        print(f"Processed {count} images", end='\r')
        pred = predictions[0]
        boxes = pred['boxes'].numpy()
        scores = pred['scores'].numpy()
        labels = pred['labels'].numpy()
        for box, score, label in zip(boxes, scores, labels):
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            coco_results.append({
                'image_id': Path(x_name[0]).stem,
                'category_id': int(label) - 1,
                'bbox': box.tolist(),
                'score': float(score)
            })

with open('lisa_mobilev3_results.json', 'w', encoding='utf-8') as f:
    json.dump(coco_results, f, indent=4)

print(f"Average inference time: {float(total_time) / count / 1e6} ms\n\n")
