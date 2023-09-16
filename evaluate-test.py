
from collections import OrderedDict
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset
from dataset import CreateMLDataset
from lisaset import LisaSet
from pytorch_faster_rcnn_tutorial.transformations import ComposeDouble
from pytorch_faster_rcnn_tutorial.transformations import (
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_double,
)

def evaluate_test(device: str):
    # root directory (project directory)
    ROOT_PATH: Path = Path(__file__).parent.absolute()

    # Change this to the path of the model checkpoint you want to evaluate
    model_path: Path = Path(ROOT_PATH) / "models/fasterrcnn_resnet50_fpn.ckpt"
    test_path: Path = ROOT_PATH / "data" / "lisav1" / "test" / "images"
    output_pathname: Path = ROOT_PATH / "output" / "lisa_resnet50_results.json"
    os.makedirs(output_pathname.parent, exist_ok=True)

    # mapping
    mapping: Dict[str, int] = LisaSet.get_labels()

    # test transformations
    transforms_test: ComposeDouble = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # dataset test
    dataset_test: Dataset = CreateMLDataset(
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
    # model: FasterRCNN = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=len(mapping) + 1)
    model: FasterRCNN = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(mapping) + 1)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
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
            if device != 'cpu':
                images = images.to(device)

            _b = time.perf_counter_ns()
            predictions = model(images)
            _e = time.perf_counter_ns()

            total_time += _e - _b
            count += 1
            print(f"Processed {count} images", end='\r')
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                coco_results.append({
                    'image_id': Path(x_name[0]).stem,
                    'category_id': int(label) - 1, # FasterRCNN adds 1 to the label, 0 is background
                    'bbox': box.tolist(),
                    'score': float(score)
                })

    with open(output_pathname, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f, indent=4)

    print(f"Average inference time: {float(total_time) / count / 1e6} ms\n\n")

if __name__ == "__main__":
    evaluate_test('cpu', 'resnet50')