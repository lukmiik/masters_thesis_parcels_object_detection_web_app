import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torchvision
from rfdetr.detr import RFDETRNano
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from ultralytics import YOLO

from src.detection import Detection

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DATASET_CLASSES = {
    "package_sorting": ["box", "mailer"],
}

MODEL_WEIGHTS = {
    "yolov11": {
        "package_sorting": (
            "training_outputs/yolov11/runs/detect/train/weights/best.pt"
        ),
    },
    "rf_detr": {
        "package_sorting": (
            "training_outputs/rf_detr/output/checkpoint_best_total.pth"
        ),
    },
    "faster_rcnn": {
        "package_sorting": (
            "training_outputs/faster_r_cnn/"
            "faster_rcnn_best_package_sorting.pth"
        ),
    },
}


class BaseDetector(ABC):
    def __init__(self, dataset: str, device: str = "auto"):
        self.dataset = dataset
        self._class_names_list = DATASET_CLASSES[dataset]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Detection:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            Detection object with boxes, scores, and class information
        """
        pass

    @property
    def class_names(self) -> list[str]:
        return self._class_names_list

    @property
    def num_classes(self) -> int:
        return len(self._class_names_list)


class YOLOv11Detector(BaseDetector):
    def __init__(
        self,
        dataset: str,
        device: str = "auto",
        confidence: float = 0.25,
        augment: bool = False,
    ):
        super().__init__(dataset, device)
        self.confidence = confidence
        self.augment = augment

        weight_path = PROJECT_ROOT / MODEL_WEIGHTS["yolov11"][dataset]
        if not weight_path.exists():
            raise FileNotFoundError(
                f"YOLOv11 weights not found: {weight_path}"
            )

        self.model = YOLO(str(weight_path))
        self.model.to(self.device)

    def predict(self, frame: np.ndarray) -> Detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(
            frame_rgb,
            verbose=False,
            conf=self.confidence,
            augment=self.augment,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        class_names = [self._class_names_list[i] for i in class_ids]

        return Detection(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=None,
        )


class RFDETRDetector(BaseDetector):
    def __init__(
        self, dataset: str, device: str = "auto", confidence: float = 0.25
    ):
        super().__init__(dataset, device)
        self.confidence = confidence

        weight_path = PROJECT_ROOT / MODEL_WEIGHTS["rf_detr"][dataset]
        if not weight_path.exists():
            raise FileNotFoundError(
                f"RF-DETR weights not found: {weight_path}"
            )

        checkpoint = torch.load(
            weight_path, map_location=self.device, weights_only=False
        )

        args = checkpoint.get("args")
        num_classes = (
            args.num_classes if args else len(self._class_names_list) + 1
        )

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        original_cwd = os.getcwd()
        try:
            os.chdir(WEIGHTS_DIR)
            self.model_wrapper = RFDETRNano(device=self.device)
        finally:
            os.chdir(original_cwd)

        self.model_wrapper.model.reinitialize_detection_head(
            num_classes=num_classes
        )

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model = cast(Any, self.model_wrapper.model.model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys in RF-DETR load: {len(missing)}")
        if unexpected:
            print(
                f"Warning: Unexpected keys in RF-DETR load: {len(unexpected)}"
            )

        model.to(self.device)
        model.eval()

        try:
            self.model_wrapper.optimize_for_inference(compile=False)
        except Exception as e:
            print(f"Warning: Failed to optimize RF-DETR for inference: {e}")

    def predict(self, frame: np.ndarray) -> Detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.model_wrapper.predict(
            frame_rgb,
            threshold=self.confidence,
        )

        if len(detections) == 0:
            return Detection(
                boxes=np.array([]).reshape(0, 4),
                scores=np.array([]),
                class_ids=np.array([], dtype=int),
                class_names=[],
                track_ids=None,
            )

        boxes = cast(
            np.ndarray,
            getattr(detections, "xyxy", np.array([]).reshape(0, 4)),
        )
        scores = cast(
            np.ndarray,
            getattr(detections, "confidence", np.array([])),
        )
        class_ids_raw = cast(
            np.ndarray | None,
            getattr(detections, "class_id", None),
        )
        if class_ids_raw is None:
            class_ids = np.array([], dtype=int)
        else:
            class_ids = class_ids_raw.astype(int)

        class_ids = class_ids - 1
        scores = scores

        valid_mask = class_ids >= 0
        if not np.all(valid_mask):
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            class_ids = class_ids[valid_mask]

        class_names = []
        for i in class_ids:
            if i < len(self._class_names_list):
                class_names.append(self._class_names_list[i])
            else:
                class_names.append(f"class_{i}")

        return Detection(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=None,
        )


class FasterRCNNDetector(BaseDetector):
    def __init__(
        self, dataset: str, device: str = "auto", confidence: float = 0.25
    ):
        super().__init__(dataset, device)
        self.confidence = confidence

        weight_path = PROJECT_ROOT / MODEL_WEIGHTS["faster_rcnn"][dataset]
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Faster R-CNN weights not found: {weight_path}"
            )

        config_path = weight_path.parent / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            num_classes = config["num_classes"]
        else:
            num_classes = len(self._class_names_list) + 1

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, num_classes=num_classes
        )

        state_dict = torch.load(
            weight_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    def predict(self, frame: np.ndarray) -> Detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = cast(torch.Tensor, self.transform(frame_rgb))
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model([image_tensor])[0]

        mask = outputs["scores"] >= self.confidence

        boxes = outputs["boxes"][mask].cpu().numpy()
        scores = outputs["scores"][mask].cpu().numpy()
        class_ids = (outputs["labels"][mask].cpu().numpy() - 1).astype(int)

        class_ids = np.clip(class_ids, 0, len(self._class_names_list) - 1)

        class_names = [self._class_names_list[i] for i in class_ids]

        return Detection(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=None,
        )


def get_detector(
    model_type: str,
    dataset: str,
    device: str = "auto",
    confidence: float = 0.25,
) -> BaseDetector:
    """
    Factory function to get the appropriate detector.

    Args:
        model_type: One of "yolov11", "rf_detr", "faster_rcnn"
        dataset: "package_sorting"
        device: Device to run on ("auto", "cuda", "cpu")
        confidence: Confidence threshold for detections

    Returns:
        Initialized detector instance
    """
    if dataset not in DATASET_CLASSES:
        raise ValueError(
            "Unknown dataset: "
            f"{dataset}. Choose from {list(DATASET_CLASSES.keys())}"
        )

    detectors = {
        "yolov11": YOLOv11Detector,
        "rf_detr": RFDETRDetector,
        "faster_rcnn": FasterRCNNDetector,
    }

    if model_type not in detectors:
        raise ValueError(
            "Unknown model type: "
            f"{model_type}. Choose from {list(detectors.keys())}"
        )

    return detectors[model_type](dataset, device, confidence)


def list_available_models() -> dict:
    available = {}
    for model_type, datasets in MODEL_WEIGHTS.items():
        available[model_type] = {}
        for dataset, weight_path in datasets.items():
            full_path = PROJECT_ROOT / weight_path
            available[model_type][dataset] = {
                "path": str(full_path),
                "exists": full_path.exists(),
            }
    return available
