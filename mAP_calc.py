# Script to calculate mAP for the test results

from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT_Path: Path = Path(__file__).parent.absolute()

cocoGt = COCO(ROOT_Path / "src/pytorch_faster_rcnn_tutorial/data/lisav1/test/coco_lisav1.json")
cocoDt = cocoGt.loadRes("lisa_mobilev3_results.json")
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()