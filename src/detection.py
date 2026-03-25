from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """Standard detection format used across all models."""

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    class_names: list[str]
    track_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.boxes)

    def filter_by_confidence(self, *, threshold: float) -> "Detection":
        mask = self.scores >= threshold
        track_ids = (
            self.track_ids[mask] if self.track_ids is not None else None
        )
        return Detection(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=[
                name
                for name, include in zip(
                    self.class_names,
                    mask,
                    strict=False,
                )
                if include
            ],
            track_ids=track_ids,
        )

    def filter_by_roi(self, *, roi: list[list[int]]) -> "Detection":
        if len(self.boxes) == 0 or not roi:
            return self

        roi_array = np.array(roi, dtype=np.int32)

        centers_x = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        centers_y = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        centers = np.column_stack((centers_x, centers_y))

        mask = np.array(
            [
                cv2.pointPolygonTest(
                    roi_array, (float(pt[0]), float(pt[1])), False
                )
                >= 0
                for pt in centers
            ]
        )

        track_ids = (
            self.track_ids[mask] if self.track_ids is not None else None
        )
        return Detection(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=[
                name
                for name, include in zip(
                    self.class_names,
                    mask,
                    strict=False,
                )
                if include
            ],
            track_ids=track_ids,
        )
