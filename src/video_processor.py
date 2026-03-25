from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import av
import cv2
import numpy as np
import supervision as sv
from numpy.typing import NDArray
from supervision.geometry.core import Point
from tqdm import tqdm

from src.detection import Detection

if TYPE_CHECKING:
    from src.inference import BaseDetector
    from src.tracking import BaseTracker, TrackerKey


def process_video(
    *,
    video_path: str | Path,
    detector: BaseDetector,
    output_path: str | Path,
    model_type: str,
    tracker_key: TrackerKey | None = None,
    tracker: BaseTracker | None = None,
    confidence_threshold: float = 0.25,
    show_progress: bool = True,
    max_frames: int | None = None,
    roi: list[list[int]] | None = None,
) -> dict:
    """
    Process a video with the given detector and save annotated output.

    Args:
        video_path: Path to input video file
        detector: Initialized detector instance
        output_path: Path for output annotated video
        confidence_threshold: Minimum confidence for displaying detections
        show_progress: Whether to show progress bar
        max_frames: Maximum number of frames to process (None for all)
        roi: Region of interest polygon as list of [x, y] coordinates
            (None for no ROI)

    Returns:
        Statistics dictionary for metrics extraction
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    output_path = output_path.with_suffix(".mp4")

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=int(round(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=30,
        color_lookup=sv.ColorLookup.TRACK,
    )

    tracker_instance = tracker
    if tracker_instance is None and tracker_key is not None:
        from src.tracking import create_tracker

        tracker_instance = create_tracker(
            model_type=model_type,
            tracker_key=tracker_key,
            detector=detector,
            frame_rate=fps if fps > 0 else 30.0,
        )

    per_frame_stats = []
    processing_times = []
    total_start_time = time.time()
    seen_track_ids: dict[str, set[int]] = {"mailer": set(), "box": set()}

    frames_iterator = range(total_frames)
    if show_progress:
        frames_iterator = tqdm(
            frames_iterator, desc="Processing video", unit="frame"
        )

    frame_id = 0
    for _ in frames_iterator:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_id >= max_frames:
            break

        start_time = time.time()
        if (
            tracker_instance is not None
            and not tracker_instance.requires_detection
        ):
            detection = tracker_instance.update(
                frame=frame,
                detection=None,
                detector=detector,
            )
        else:
            detection = detector.predict(frame)

        if confidence_threshold > 0:
            detection = detection.filter_by_confidence(
                threshold=confidence_threshold
            )

        if roi is not None:
            detection = detection.filter_by_roi(roi=roi)

        if (
            tracker_instance is not None
            and tracker_instance.requires_detection
        ):
            detection = tracker_instance.update(
                frame=frame,
                detection=detection,
                detector=detector,
            )

        inference_time_ms = (time.time() - start_time) * 1000
        processing_times.append(inference_time_ms)

        if len(detection) > 0:
            sv_detections = sv.Detections(
                xyxy=detection.boxes,
                confidence=detection.scores,
                class_id=detection.class_ids,
                tracker_id=detection.track_ids,
            )

            if detection.track_ids is None:
                track_ids = [None] * len(detection.class_names)
            else:
                track_ids = detection.track_ids.tolist()

            labels = []
            for name, conf, track_id in zip(
                detection.class_names,
                detection.scores,
                track_ids,
                strict=False,
            ):
                if track_id is None:
                    labels.append(f"{name} {conf:.2f}")
                else:
                    labels.append(f"{name} {conf:.2f} ID:{int(track_id)}")

            if detection.track_ids is not None:
                frame = trace_annotator.annotate(frame, sv_detections)
            frame = box_annotator.annotate(frame, sv_detections)
            frame = label_annotator.annotate(
                frame, sv_detections, labels=labels
            )

        class_counts = {}
        class_confidences = {}
        for name, conf in zip(
            detection.class_names,
            detection.scores,
            strict=False,
        ):
            class_counts[name] = class_counts.get(name, 0) + 1
            if name not in class_confidences:
                class_confidences[name] = []
            class_confidences[name].append(float(conf))

        current_mailers = class_counts.get("mailer", 0)
        current_boxes = class_counts.get("box", 0)

        if detection.track_ids is not None:
            for name, track_id in zip(
                detection.class_names, detection.track_ids, strict=False
            ):
                if track_id is not None and name in seen_track_ids:
                    seen_track_ids[name].add(int(track_id))

        total_mailers = len(seen_track_ids["mailer"])
        total_boxes = len(seen_track_ids["box"])

        current_text = (
            f"Current: {current_mailers} mailers, {current_boxes} boxes"
        )
        total_text = f"Total: {total_mailers} mailers, {total_boxes} boxes"

        frame = sv.draw_text(
            scene=frame,
            text=current_text,
            text_anchor=Point(x=220, y=30),
            text_color=sv.Color(r=255, g=255, b=255),
            text_scale=0.8,
            text_thickness=2,
            text_padding=10,
            background_color=sv.Color(r=0, g=0, b=0),
        )
        frame = sv.draw_text(
            scene=frame,
            text=total_text,
            text_anchor=Point(x=205, y=70),
            text_color=sv.Color(r=255, g=255, b=255),
            text_scale=0.8,
            text_thickness=2,
            text_padding=10,
            background_color=sv.Color(r=0, g=0, b=0),
        )

        if roi is not None:
            roi_array = np.array(roi, dtype=np.int32)
            cv2.polylines(
                frame,
                [roi_array],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_uint8: NDArray[np.uint8] = frame_rgb.astype(np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb_uint8, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

        avg_conf_per_class = {
            name: sum(confs) / len(confs)
            for name, confs in class_confidences.items()
        }

        frame_stats = {
            "frame_id": frame_id,
            "timestamp_ms": (frame_id / fps) * 1000,
            "inference_time_ms": round(inference_time_ms, 2),
            "num_detections": len(detection),
            "class_counts": class_counts,
            "avg_confidence_per_class": {
                k: round(v, 3) for k, v in avg_conf_per_class.items()
            },
            "detections": [
                {
                    "class": name,
                    "confidence": round(float(conf), 3),
                    "bbox": [round(float(x), 1) for x in box],
                }
                for name, conf, box in zip(
                    detection.class_names,
                    detection.scores,
                    detection.boxes,
                    strict=False,
                )
            ],
        }
        per_frame_stats.append(frame_stats)

        frame_id += 1

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    cap.release()

    total_processing_time = time.time() - total_start_time

    total_detections = sum(s["num_detections"] for s in per_frame_stats)

    class_distribution = {}
    all_class_confidences = {}
    for stats in per_frame_stats:
        for class_name, count in stats["class_counts"].items():
            class_distribution[class_name] = (
                class_distribution.get(class_name, 0) + count
            )
        for det in stats["detections"]:
            name = det["class"]
            if name not in all_class_confidences:
                all_class_confidences[name] = []
            all_class_confidences[name].append(det["confidence"])

    avg_confidence_per_class = {
        name: round(sum(confs) / len(confs), 3)
        for name, confs in all_class_confidences.items()
    }

    avg_inference_time_ms = (
        round(sum(processing_times) / len(processing_times), 2)
        if processing_times
        else 0
    )
    avg_inference_fps = (
        1000.0 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0
    )

    return {
        "model": detector.__class__.__name__.replace("Detector", "").lower(),
        "tracker": tracker_key if tracker_key is not None else "none",
        "dataset": detector.dataset,
        "video_file": video_path.name,
        "output_file": output_path.name,
        "total_frames": frame_id,
        "video_fps": round(fps, 2),
        "avg_inference_fps": round(avg_inference_fps, 2),
        "processing_time_seconds": round(total_processing_time, 2),
        "avg_inference_time_ms": avg_inference_time_ms,
        "per_frame_stats": per_frame_stats,
        "aggregate_stats": {
            "total_detections": total_detections,
            "class_distribution": class_distribution,
            "avg_confidence_per_class": avg_confidence_per_class,
            "detection_density_per_frame": round(
                total_detections / frame_id, 2
            )
            if frame_id > 0
            else 0,
        },
    }


def process_single_frame(
    *,
    frame: np.ndarray,
    detector: BaseDetector,
    model_type: str,
    tracker_key: TrackerKey | None = None,
    tracker: BaseTracker | None = None,
    confidence_threshold: float = 0.25,
) -> tuple[np.ndarray, Detection]:
    """
    Process a single frame and return the annotated frame and detections.

    Args:
        frame: Input BGR image
        detector: Initialized detector instance
        confidence_threshold: Minimum confidence for displaying detections

    Returns:
        Tuple of (annotated_frame, detection)
    """
    tracker_instance = tracker
    if tracker_instance is None and tracker_key is not None:
        from src.tracking import create_tracker

        tracker_instance = create_tracker(
            model_type=model_type,
            tracker_key=tracker_key,
            detector=detector,
            frame_rate=30.0,
        )

    if (
        tracker_instance is not None
        and not tracker_instance.requires_detection
    ):
        detection = tracker_instance.update(
            frame=frame,
            detection=None,
            detector=detector,
        )
    else:
        detection = detector.predict(frame)

    if confidence_threshold > 0:
        detection = detection.filter_by_confidence(
            threshold=confidence_threshold
        )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=30,
        color_lookup=sv.ColorLookup.TRACK,
    )

    annotated_frame = frame.copy()
    if tracker_instance is not None and tracker_instance.requires_detection:
        detection = tracker_instance.update(
            frame=frame,
            detection=detection,
            detector=detector,
        )

    if len(detection) > 0:
        sv_detections = sv.Detections(
            xyxy=detection.boxes,
            confidence=detection.scores,
            class_id=detection.class_ids,
            tracker_id=detection.track_ids,
        )

        if detection.track_ids is None:
            track_ids = [None] * len(detection.class_names)
        else:
            track_ids = detection.track_ids.tolist()

        labels = []
        for name, conf, track_id in zip(
            detection.class_names,
            detection.scores,
            track_ids,
            strict=False,
        ):
            if track_id is None:
                labels.append(f"{name} {conf:.2f}")
            else:
                labels.append(f"{name} {conf:.2f} ID:{int(track_id)}")

        if detection.track_ids is not None:
            annotated_frame = trace_annotator.annotate(
                annotated_frame, sv_detections
            )
        annotated_frame = box_annotator.annotate(
            annotated_frame, sv_detections
        )
        annotated_frame = label_annotator.annotate(
            annotated_frame, sv_detections, labels=labels
        )

    return annotated_frame, detection
