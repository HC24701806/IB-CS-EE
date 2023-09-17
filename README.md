# IB CS EE
Code for my IB Extended Essay. In this paper I am comparing the accuracy and speed of Faster R-CNN and Yolov8 in traffic sign detection.

## Set Up
1. Install required packaged
    ```pip install -r requirements.txt```
2. Configure Neptune logger <br>
   Training code uses Neptune logger. The API token needs to be put into .env file in the root folder.

## Data Set
https://universe.roboflow.com/dakota-smith/lisa-road-signs

Downloaded the [CreateML JSON format](https://roboflow.com/formats/createml-json). 

Faster R-CNN: Wrote a custom dataset to read from it. 

YOLO: Script to convert data into [Yolo format](https://docs.ultralytics.com/datasets/detect/).

## Faster RCNN and Training code
Based on [the code](https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial) from [Johannes Schmidt's "Train your own object detector with Faster-RCNN & PyTorch"](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70).

However, I used the pre-implemented models in Torchvision instead. There are two models used:
- fasterrcnn_mobilenet_v3_large_fpn
- fasterrcnn_resnet50_fpn_v2

## Yolo v8
Here I use the existing ultralytics.yolo package.

To train:

    ``` yolo train data=data/lisav1/lisa-train.yaml model=yolov8s.pt epochs=150```

To evaluate:

    ``` yolo val data=data/lisav1/lisa-test.yaml model=<best.pt> save_json=true```

prediction.json could be used in evaluation of mAP metrics.

## Metrics
Used the pycocotools to calculate mAP metrics

## Files
- generate_coco_gt.py: to generate ground truth of lisa dataset to for COCO API.
- evaluate-test.py: to run inference of Faster R-CNN and generate result json for COCO API.
- train_yolo.py: to train Yolo.
- train_fasterrcnn.py: to train Faster R-CNN.

## Cite this work
```
@software{Cong_Comparing_Faster_RCNN_2023,
author = {Cong, Haolin},
month = sep,
title = {{Comparing Faster RCNN and Yolo v8 in Traffic Sign Detection}},
url = {https://github.com/HC24701806/IB-CS-EE},
version = {1.0.0},
year = {2023}
}
```
