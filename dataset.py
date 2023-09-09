import csv
import pathlib
import torch
from torch.utils.data import Dataset
from skimage.color import rgba2rgb
from skimage.io import imread
from typing import Dict
from pytorch_faster_rcnn_tutorial.transformations import ComposeDouble,map_class_to_int

class CreateMLDataset(Dataset):
    def __init__(
        self,
        inputPath : pathlib.Path,
        transform: ComposeDouble = None,
        use_cache: bool = False,
        convert_to_format: str = None,
        mapping: Dict = None,
        ) -> None:
        super().__init__()

        self.root = inputPath
        self.transform = transform
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        itemsMap = {}

        with open(inputPath / "_annotations.csv", newline='') as f:
            annocsv = csv.reader(f)
            next(annocsv)
            for filename, width, height, label, xmin, ymin, xmax, ymax in annocsv:
                width = float(width)
                height = float(height)
                xmin = float(xmin)
                ymin = float(ymin)
                xmax = float(xmax)
                ymax = float(ymax)
                assert(xmin >=0 and xmin <= xmax and xmax <= width)
                assert(ymin >=0 and ymin <= ymax and ymax <= height)
                assert((inputPath / filename).exists())
                if filename in itemsMap:
                    entry = itemsMap[filename]
                    entry['boxes'].append([xmin, ymin, xmax, ymax])
                    entry['labels'].append(label)
                else:
                    itemsMap[filename] = {"boxes": [[xmin, ymin, xmax, ymax]],
                                            "labels": [label]}
        self.items = list(itemsMap.items())

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: int):
        fn, y = self.items[index]
        x = imread(self.root / fn)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Read boxes
        try:
            boxes = torch.from_numpy(y["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y["boxes"]).to(torch.float32)

        # Read scores
        if "scores" in y.keys():
            try:
                scores = torch.from_numpy(y["scores"]).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y["scores"]).to(torch.float32)

        # Label Mapping
        if self.mapping:
            labels = map_class_to_int(y["labels"], mapping=self.mapping)
        else:
            labels = y["labels"]

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Create target
        target = {"boxes": boxes, "labels": labels}

        if "scores" in y.keys():
            target["scores"] = scores

        # Preprocessing
        target = {
            key: value.numpy() for key, value in target.items()
        }  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {
            key: torch.from_numpy(value).type(torch.int64)
            for key, value in target.items()
        }

        return {
            "x": x,
            "y": target,
            "x_name": fn,
            "y_name": fn,
        }