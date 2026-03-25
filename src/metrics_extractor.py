import csv
import json
from pathlib import Path
from typing import Any


def save_metrics(
    stats: dict,
    output_path: str | Path,
    include_per_frame: bool = True,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not include_per_frame:
        stats = {k: v for k, v in stats.items() if k != "per_frame_stats"}

    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_metrics(json_path: str | Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _compute_confidence_metrics(metrics_list: list[dict]) -> dict:
    all_classes: set[str] = set()
    for stats in metrics_list:
        class_dist = stats.get("aggregate_stats", {}).get(
            "class_distribution", {}
        )
        all_classes.update(class_dist.keys())
    return {"all_classes": sorted(all_classes)}


def create_comparison_csv(
    metrics_list: list[dict],
    output_path: str | Path,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    class_info = _compute_confidence_metrics(metrics_list)
    all_classes = class_info["all_classes"]

    fieldnames = [
        "model",
        "tracker",
        "dataset",
        "video_file",
        "total_frames",
        "video_fps",
        "avg_inference_fps",
        "processing_time_seconds",
        "avg_inference_time_ms",
        "total_detections",
        "detection_density_per_frame",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
    ] + all_classes

    with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in metrics_list:
            agg = stats.get("aggregate_stats", {})
            class_dist = agg.get("class_distribution", {})

            confidences: list[float] = []
            for frame in stats.get("per_frame_stats", []):
                for det in frame.get("detections", []):
                    if "confidence" in det:
                        confidences.append(det["confidence"])

            row: dict[str, Any] = {
                "model": stats.get("model", ""),
                "tracker": stats.get("tracker", "none"),
                "dataset": stats.get("dataset", ""),
                "video_file": stats.get("video_file", ""),
                "total_frames": stats.get("total_frames", 0),
                "video_fps": stats.get("video_fps", 0),
                "avg_inference_fps": stats.get("avg_inference_fps", 0),
                "processing_time_seconds": stats.get(
                    "processing_time_seconds", 0
                ),
                "avg_inference_time_ms": stats.get("avg_inference_time_ms", 0),
                "total_detections": agg.get("total_detections", 0),
                "detection_density_per_frame": agg.get(
                    "detection_density_per_frame", 0
                ),
                "avg_confidence": (
                    round(sum(confidences) / len(confidences), 3)
                    if confidences
                    else 0
                ),
                "min_confidence": (
                    round(min(confidences), 3) if confidences else 0
                ),
                "max_confidence": (
                    round(max(confidences), 3) if confidences else 0
                ),
            }

            for class_name in all_classes:
                row[class_name] = class_dist.get(class_name, 0)

            writer.writerow(row)


def create_class_distribution_csv(
    metrics_list: list[dict],
    output_path: str | Path,
) -> None:
    """
    Create a CSV with class distribution across all model/dataset combinations.

    Args:
        metrics_list: List of statistics dictionaries
        output_path: Path for output CSV file
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    all_classes = set()
    for stats in metrics_list:
        class_dist = stats.get("aggregate_stats", {}).get(
            "class_distribution", {}
        )
        all_classes.update(class_dist.keys())

    all_classes = sorted(all_classes)

    fieldnames = ["model", "dataset"] + list(all_classes) + ["total"]

    with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in metrics_list:
            class_dist = stats.get("aggregate_stats", {}).get(
                "class_distribution", {}
            )
            row = {
                "model": stats.get("model", ""),
                "dataset": stats.get("dataset", ""),
                "total": sum(class_dist.values()),
            }
            for class_name in all_classes:
                row[class_name] = class_dist.get(class_name, 0)
            writer.writerow(row)


def summarize_metrics(stats: dict) -> str:
    """
    Create a human-readable summary of metrics.

    Args:
        stats: Statistics dictionary from process_video()

    Returns:
        Formatted summary string
    """
    agg = stats.get("aggregate_stats", {})

    lines = [
        f"Model: {stats.get('model', 'unknown')}",
        f"Dataset: {stats.get('dataset', 'unknown')}",
        f"Video: {stats.get('video_file', 'unknown')}",
        "",
        "Performance:",
        f"  - Total frames: {stats.get('total_frames', 0)}",
        f"  - Video FPS: {stats.get('video_fps', 0):.1f}",
        f"  - Inference FPS: {stats.get('avg_inference_fps', 0):.1f}",
        "  - Avg inference time: "
        f"{stats.get('avg_inference_time_ms', 0):.1f} ms",
        "  - Total processing time: "
        f"{stats.get('processing_time_seconds', 0):.1f} s",
        "",
        "Detection Statistics:",
        f"  - Total detections: {agg.get('total_detections', 0)}",
        "  - Detections per frame: "
        f"{agg.get('detection_density_per_frame', 0):.2f}",
        "",
        "Class Distribution:",
    ]

    class_dist = agg.get("class_distribution", {})
    for class_name, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        lines.append(f"  - {class_name}: {count}")

    return "\n".join(lines)


def get_aggregate_summary(metrics_list: list[dict]) -> dict:
    summary: dict[str, Any] = {
        "total_videos_processed": len(metrics_list),
        "models": list(set(s.get("model", "") for s in metrics_list)),
        "datasets": list(set(s.get("dataset", "") for s in metrics_list)),
        "by_model": {},
        "by_dataset": {},
    }

    for stats in metrics_list:
        model = str(stats.get("model", "unknown"))
        by_model = summary["by_model"]
        if model not in by_model:
            by_model[model] = {
                "avg_fps": [],
                "avg_inference_time_ms": [],
                "total_detections": 0,
            }
        by_model[model]["avg_fps"].append(stats.get("avg_inference_fps", 0))
        by_model[model]["avg_inference_time_ms"].append(
            stats.get("avg_inference_time_ms", 0)
        )
        by_model[model]["total_detections"] += stats.get(
            "aggregate_stats", {}
        ).get("total_detections", 0)

    for _model, data in summary["by_model"].items():
        data["avg_fps"] = (
            sum(data["avg_fps"]) / len(data["avg_fps"])
            if data["avg_fps"]
            else 0
        )
        data["avg_inference_time_ms"] = (
            sum(data["avg_inference_time_ms"])
            / len(data["avg_inference_time_ms"])
            if data["avg_inference_time_ms"]
            else 0
        )

    return summary
