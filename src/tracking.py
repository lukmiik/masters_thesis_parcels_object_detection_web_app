from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, SORTTracker

from src.detection import Detection

if TYPE_CHECKING:
    from src.inference import BaseDetector, YOLOv11Detector

TrackerKey = Literal["botsort", "bytetrack", "sort"]
TrackerBackend = Literal["yolo_native", "roboflow"]


@dataclass(frozen=True)
class TrackerInfo:
    key: TrackerKey
    name: str
    backend: TrackerBackend
    yolo_config: str | None


TRACKER_LABELS: dict[TrackerKey, str] = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "sort": "SORT",
}

YOLO_NATIVE_TRACKERS: dict[TrackerKey, str] = {
    "botsort": "botsort.yaml",
    "bytetrack": "bytetrack.yaml",
}

MODEL_TRACKERS: dict[str, list[TrackerKey]] = {
    "yolov11": ["botsort", "bytetrack"],
    "rf_detr": ["sort", "bytetrack"],
    "faster_rcnn": ["sort", "bytetrack"],
}

DEFAULT_TRACKER_BY_MODEL: dict[str, TrackerKey] = {
    "yolov11": "bytetrack",
    "rf_detr": "bytetrack",
    "faster_rcnn": "bytetrack",
}


def list_tracker_keys(*, model_type: str) -> list[TrackerKey]:
    return list(MODEL_TRACKERS.get(model_type, []))


def list_all_tracker_keys() -> list[TrackerKey]:
    keys: set[TrackerKey] = set()
    for trackers in MODEL_TRACKERS.values():
        keys.update(trackers)
    return sorted(keys)


def get_tracker_display_name(*, tracker_key: TrackerKey) -> str:
    return TRACKER_LABELS.get(tracker_key, tracker_key)


def get_default_tracker(*, model_type: str) -> TrackerKey:
    if model_type not in DEFAULT_TRACKER_BY_MODEL:
        raise ValueError(f"No default tracker configured for {model_type}")
    return DEFAULT_TRACKER_BY_MODEL[model_type]


def validate_tracker_for_model(
    *, model_type: str, tracker_key: TrackerKey
) -> None:
    valid = MODEL_TRACKERS.get(model_type, [])
    if tracker_key not in valid:
        raise ValueError(
            f"Tracker '{tracker_key}' not supported for {model_type}. "
            f"Choose from {valid}"
        )


def _to_sv_detections(*, detection: Detection) -> sv.Detections:
    tracker_id = (
        detection.track_ids.astype(int)
        if detection.track_ids is not None
        else None
    )
    return sv.Detections(
        xyxy=detection.boxes,
        confidence=detection.scores,
        class_id=detection.class_ids,
        tracker_id=tracker_id,
    )


def _detections_from_sv(
    *, detections: sv.Detections, class_names: list[str]
) -> Detection:
    boxes = detections.xyxy
    scores = detections.confidence
    class_ids = detections.class_id
    tracker_ids = detections.tracker_id

    if boxes is None or len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)

    if scores is None or len(scores) != len(boxes):
        scores = np.zeros(len(boxes), dtype=float)

    if class_ids is None or len(class_ids) != len(boxes):
        class_ids = np.array([], dtype=int)
    else:
        class_ids = class_ids.astype(int)

    if tracker_ids is None or len(tracker_ids) != len(boxes):
        tracker_ids = None
    else:
        tracker_ids = tracker_ids.astype(int)

    resolved_names: list[str] = []
    for class_id in class_ids:
        if class_id < len(class_names):
            resolved_names.append(class_names[class_id])
        else:
            resolved_names.append(f"class_{class_id}")

    return Detection(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        class_names=resolved_names,
        track_ids=tracker_ids,
    )


class BaseTracker(ABC):
    @property
    @abstractmethod
    def requires_detection(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        *,
        frame: np.ndarray,
        detection: Detection | None,
        detector: BaseDetector,
    ) -> Detection:
        raise NotImplementedError


class RoboflowTrackerAdapter(BaseTracker):
    def __init__(
        self,
        *,
        tracker_key: TrackerKey,
        class_names: list[str],
        frame_rate: float,
    ) -> None:
        if tracker_key == "bytetrack":
            self._tracker = ByteTrackTracker(frame_rate=frame_rate)
        elif tracker_key == "sort":
            self._tracker = SORTTracker(frame_rate=frame_rate)
        else:
            raise ValueError(f"Unsupported roboflow tracker: {tracker_key}")

        self._class_names = class_names

    @property
    def requires_detection(self) -> bool:
        return True

    def update(
        self,
        *,
        frame: np.ndarray,
        detection: Detection | None,
        detector: BaseDetector,
    ) -> Detection:
        if detection is None:
            raise ValueError("Detection is required for roboflow trackers")

        sv_detections = _to_sv_detections(detection=detection)
        tracked = self._tracker.update(sv_detections)
        return _detections_from_sv(
            detections=tracked,
            class_names=self._class_names,
        )


class YoloNativeTrackerAdapter(BaseTracker):
    def __init__(
        self,
        *,
        tracker_key: TrackerKey,
        detector: YOLOv11Detector,
    ) -> None:
        if tracker_key not in YOLO_NATIVE_TRACKERS:
            raise ValueError(f"Unsupported YOLO tracker: {tracker_key}")

        self._tracker_config = YOLO_NATIVE_TRACKERS[tracker_key]
        self._detector = detector

    @property
    def requires_detection(self) -> bool:
        return False

    def update(
        self,
        *,
        frame: np.ndarray,
        detection: Detection | None,
        detector: BaseDetector,
    ) -> Detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.model.track(
            frame_rgb,
            verbose=False,
            conf=self._detector.confidence,
            augment=self._detector.augment,
            tracker=self._tracker_config,
            persist=True,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return Detection(
                boxes=np.array([]).reshape(0, 4),
                scores=np.array([]),
                class_ids=np.array([], dtype=int),
                class_names=[],
                track_ids=None,
            )

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            results.boxes.id.int().cpu().numpy()
            if results.boxes.id is not None
            else None
        )

        class_names = [
            self._detector.class_names[i]
            if i < len(self._detector.class_names)
            else f"class_{i}"
            for i in class_ids
        ]

        return Detection(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=track_ids,
        )


def create_tracker(
    *,
    model_type: str,
    tracker_key: TrackerKey,
    detector: BaseDetector,
    frame_rate: float,
) -> BaseTracker:
    validate_tracker_for_model(model_type=model_type, tracker_key=tracker_key)

    if model_type == "yolov11" and tracker_key in YOLO_NATIVE_TRACKERS:
        from src.inference import YOLOv11Detector

        if not isinstance(detector, YOLOv11Detector):
            raise ValueError("YOLO native tracking requires YOLOv11 detector")
        return YoloNativeTrackerAdapter(
            tracker_key=tracker_key,
            detector=detector,
        )

    return RoboflowTrackerAdapter(
        tracker_key=tracker_key,
        class_names=detector.class_names,
        frame_rate=frame_rate,
    )
