# Script to calculate mAP for the test results

from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import numpy as np

ROOT_Path: Path = Path(__file__).parent.absolute()

cocoGt = COCO(ROOT_Path / "data/lisav1/test/coco_lisav1.json")
cocoDtRS50 = cocoGt.loadRes("output/lisa_resnet50_results.json")
cocoEval = COCOeval(cocoGt, cocoDtRS50, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
pc_res50 = cocoEval.eval["precision"].copy()
pc_res50 = pc_res50[0, :, :, 0, 2] # 0.5 IoU, all recall, all cat, all areas, 100 maxDets
mask = (pc_res50 < 0).any(axis=0)
pc_res50 = np.mean(pc_res50[:, ~mask], axis=1)
print("mAP for ResNet50:")
cocoEval.summarize()
print("----------------------------------------\n")

cocoDtM3 = cocoGt.loadRes("output/lisa_mobilenetv3_results.json")
cocoEval = COCOeval(cocoGt, cocoDtM3, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
pc_m3 = cocoEval.eval["precision"].copy()
pc_m3 = pc_m3[0, :, :, 0, 2] # 0.5 IoU, all recall, all cat, all areas, 100 maxDets
mask = (pc_m3 < 0).any(axis=0)
pc_m3 = np.mean(pc_m3[:, ~mask], axis=1)
print("mAP for ResNet50:")
cocoEval.summarize()
print("----------------------------------------\n")

cocoDtYolo = cocoGt.loadRes("output/predictions.json")
cocoEval = COCOeval(cocoGt, cocoDtYolo, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
pc_yolo = cocoEval.eval["precision"].copy()
pc_yolo = pc_yolo[0, :, :, 0, 2] # 0.5 IoU, all recall, all cat, all areas, 100 maxDets
mask = (pc_yolo < 0).any(axis=0)
pc_yolo = np.mean(pc_yolo[:, ~mask], axis=1)
print("mAP for ResNet50:")
cocoEval.summarize()
print("----------------------------------------\n")

recall = np.linspace(0, 1, 101)

plt.figure()
plt.plot(recall, pc_res50, label="Faster RCNN ResNet50", color="red")
plt.plot(recall, pc_m3, label="Faster RCNN MobileNetV3", color="blue")
plt.plot(recall, pc_yolo, label="YOLOv8s", color="green")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="center right")
plt.show()

plt.figure()
