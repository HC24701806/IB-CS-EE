# Description: Generate COCO ground truth from LISA dataset
# https://cocodataset.org/#format-data

import json
import csv
from pathlib import Path

ROOT_PATH: Path = Path(__file__).parent.absolute() / "src" / "pytorch_faster_rcnn_tutorial" / "data" / "lisav1"

categories = {}

for sub in ["train", "valid", "test"]:
    images = {}
    annotations = []
    folder = ROOT_PATH / sub
    with open(folder / "images" / "_annotations.csv", "r", encoding='utf-8') as f:
        cv = csv.reader(f)
        next(cv)
        for filename,width,height,label,xmin,ymin,xmax,ymax in cv:
            width = int(width)
            height = int(height)
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            assert(xmin >=0 and xmin <= xmax and xmax <= width)
            assert(ymin >=0 and ymin <= ymax and ymax <= height)
            assert((folder / "images" / filename).exists())
            if filename not in images:
                images[filename] = {'id': Path(filename).stem, 'width': width, 'height': height, 'file_name': filename}
            if label not in categories:
                categories[label] = len(categories)
            annotations.append({
                'id': len(annotations) + 1,
                'image_id': images[filename]['id'],
                'category_id': categories[label],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0,
            })

    coco_gt = {
        'info': {
            'description': 'LISA Traffic Sign Dataset',
            'version': '1.0',
            'url': 'https://universe.roboflow.com/dakota-smith/lisa-road-signs',
        },
        'images': list(images.values()),
        'annotations': annotations,
        'categories': [{'id': v, 'name': k} for k, v in categories.items()],
    }

    with open(folder / 'coco_lisav1.json', 'w', encoding='utf-8') as f:
        json.dump(coco_gt, f, indent=4, sort_keys=False)
