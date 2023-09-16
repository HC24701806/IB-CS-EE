import logging
import pathlib
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

import albumentations
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights

from pytorch_faster_rcnn_tutorial.faster_RCNN import (
    FasterRCNNLightning,
)

from pytorch_faster_rcnn_tutorial.transformations import (
    AlbumentationWrapper,
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import collate_double
from dataset import CreateMLDataset
from lisaset import LisaSet

logger: logging.Logger = logging.getLogger(__name__)

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# root directory (project directory)
ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.absolute()


class NeptuneSettings():
    """
    Reads the variables from .env file.
    Errors will be raised if the required variables are not set.
    """
    PROJECT: str = "haolin.cong/TrafficSign-Yolo"
    def __init__(self):
        with open(pathlib.Path(__file__).parent / '.env', 'r') as f:
            for line in f.readlines():
                key, value = line.strip().split('=', maxsplit=1)
                if key == 'NEPTUNE_API_TOKEN':
                    self.api_key = value

# mapping
mapping: Dict[str, int] = LisaSet.get_labels()
mapping.update((k, v + 1) for k, v in mapping.items()) # add background class

@dataclass
class Parameters:
    """
    Dataclass for the parameters.
    """

    BATCH_SIZE: int = 16
    CACHE: bool = True
    SAVE_DIR: Optional[
        str
    ] = None  # checkpoints will be saved to cwd (current working directory) if None
    LOG_MODEL: bool = False  # whether to log the model to neptune after training
    ACCELERATOR: Optional[str] = "auto"  # set to "gpu" if you want to use GPU
    LR: float = 0.001
    PRECISION: int = 32
    CLASSES: int = len(mapping) + 1
    SEED: int = 42
    MAXEPOCHS: int = 150
    PATIENCE: int = 50
    FPN: bool = False
    ANCHOR_SIZE: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),)
    ASPECT_RATIOS: Tuple[Tuple[float, ...]] = ((0.5, 1.0, 2.0),)
    MIN_SIZE: int = 1024
    MAX_SIZE: int = 1025
    IMG_MEAN: List = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMG_STD: List = field(default_factory=lambda: [0.229, 0.224, 0.225])
    IOU_THRESHOLD: float = 0.5
    FAST_DEV_RUN: bool = False

    def __post_init__(self):
        if self.SAVE_DIR is None:
            self.SAVE_DIR: str = str(pathlib.Path.cwd())

def get_fasterrcnn_mobilenet_v3_large_fpn(num_classes: int) -> FasterRCNN:
    model = fasterrcnn_mobilenet_v3_large_fpn(
        num_classes=num_classes,
    )

    # these are not necessary however used in the tutorial code
    model.num_classes = num_classes
    model.image_mean = [0.485, 0.456, 0.406]
    model.image_std = [0.229, 0.224, 0.225]
    return model

def get_fasterrcnn_resnet50_fpn_default(num_classes: int) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn_v2(
        num_classes=num_classes,
        weight_backbone=ResNet50_Weights.IMAGENET1K_V2,
    )
    model.num_classes = num_classes
    model.image_mean = [0.485, 0.456, 0.406]
    model.image_std = [0.229, 0.224, 0.225]
    return model

def train():
    # environment variables (pydantic BaseSettings class)
    neptune_settings: NeptuneSettings = NeptuneSettings()

    # parameters (dataclass)
    parameters: Parameters = Parameters()

    # data path relative to this file (pathlib)
    train_path: pathlib.Path = (
        ROOT_PATH / "data" / "lisav1" / "train" / "images"
    )
    valid_path: pathlib.Path = (
        ROOT_PATH / "data" / "lisav1" / "valid" / "images"
    )
    test_path: pathlib.Path = (
        ROOT_PATH / "data" / "lisav1" / "test" / "images"
    )

    # training transformations and augmentations
    transforms_training: ComposeDouble = ComposeDouble(
        [
            Clip(),
            AlbumentationWrapper(albumentation=albumentations.HorizontalFlip(p=0.5)),
            AlbumentationWrapper(
                albumentation=albumentations.RandomScale(p=0.5, scale_limit=0.5)
            ),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # validation transformations
    transforms_validation: ComposeDouble = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # test transformations
    transforms_test: ComposeDouble = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # random seed (function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random)
    seed_everything(parameters.SEED)

    # dataset training
    dataset_train: Dataset = CreateMLDataset(
        inputPath=train_path,
        transform=transforms_training,
        convert_to_format=None,
        mapping=mapping,
    )

    # dataset validation
    dataset_valid: Dataset = CreateMLDataset(
        inputPath=valid_path,
        transform=transforms_validation,
        convert_to_format=None,
        mapping=mapping,
    )

    # dataset test
    dataset_test: Dataset = CreateMLDataset(
        inputPath=test_path,
        transform=transforms_test,
        convert_to_format=None,
        mapping=mapping,
    )

    # dataloader training
    dataloader_train: DataLoader = DataLoader(
        dataset=dataset_train,
        batch_size=parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_double,
    )

    # dataloader validation
    dataloader_valid: DataLoader = DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # dataloader test
    dataloader_test: DataLoader = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # neptune logger (neptune-client)
    neptune_logger: NeptuneLogger = NeptuneLogger(
        api_key=neptune_settings.api_key,
        project=f"{neptune_settings.PROJECT}",  # use your neptune name here
        name="Faster RCNN",
        log_model_checkpoints=False,
    )

    # log hyperparameters
    neptune_logger.log_hyperparams(asdict(parameters))

    # model init
    model: FasterRCNN = get_fasterrcnn_mobilenet_v3_large_fpn(parameters.CLASSES)

    # lightning model
    model: FasterRCNNLightning = FasterRCNNLightning(
        model=model, lr=parameters.LR, iou_threshold=parameters.IOU_THRESHOLD
    )

    # callbacks
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="Validation_mAP", mode="max"
    )
    learning_rate_callback: LearningRateMonitor = LearningRateMonitor(
        logging_interval="step", log_momentum=False
    )
    early_stopping_callback: EarlyStopping = EarlyStopping(
        monitor="Validation_mAP", patience=parameters.PATIENCE, mode="max"
    )

    # trainer init
    trainer: Trainer = Trainer(
        accelerator=parameters.ACCELERATOR,
        logger=neptune_logger,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
            early_stopping_callback,
        ],
        default_root_dir=parameters.SAVE_DIR,  # where checkpoints are saved to
        log_every_n_steps=1,  # increase to reduce the amount of log flushes (lowers the overhead)
        num_sanity_val_steps=0,  # set to 0 to skip sanity check
        max_epochs=parameters.MAXEPOCHS,
        fast_dev_run=parameters.FAST_DEV_RUN,  # set to True to test the pipeline with one batch and without validation, testing and logging
    )

    # start training
    trainer.fit(
        model=model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_valid,
    )

    if not parameters.FAST_DEV_RUN:
        # start testing
        trainer.test(ckpt_path="best", dataloaders=dataloader_test)

    # stop logger
    neptune_logger.experiment.stop()
    logger.info("Training finished")
    print(f"model saved at {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
