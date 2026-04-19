import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import streamlit as st

from src.metrics_extractor import save_metrics, summarize_metrics
from src.paths import (
    INPUT_VIDEOS_DIR,
    LABELED_VIDEOS_DIR,
    METRICS_DIR,
    ROI_FILE,
)
from src.tracking import (
    MODEL_TRACKERS,
    TRACKER_LABELS,
    TrackerKey,
)

MODELS: dict[str, str] = {
    "yolov11": "YOLOv11",
    "rf_detr": "RF-DETR",
    "faster_rcnn": "Faster R-CNN",
}

TRACKER_DISPLAY: dict[str, str] = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "sort": "SORT",
}


@dataclass
class JobResult:
    model: str
    tracker: str
    video: str
    success: bool
    output_video: Path | None
    output_metrics: Path | None
    summary: str
    error: str | None


def load_roi_data() -> dict[str, list[list[int]]]:
    if not ROI_FILE.exists():
        return {}
    try:
        with open(ROI_FILE) as f:
            data = json.load(f)
        return {video_name: info["roi"] for video_name, info in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def get_input_videos() -> dict[str, Path]:
    videos: dict[str, Path] = {}
    if INPUT_VIDEOS_DIR.exists():
        for video_path in sorted(INPUT_VIDEOS_DIR.glob("*.mp4")):
            videos[video_path.stem] = video_path
    return videos


def build_selection_options() -> tuple[list[str], dict[str, tuple[str, str]]]:
    options: list[str] = []
    mapping: dict[str, tuple[str, str]] = {}

    for model_key, model_name in MODELS.items():
        trackers = MODEL_TRACKERS.get(model_key, [])
        for tracker_key in trackers:
            tracker_label = TRACKER_LABELS.get(tracker_key, tracker_key)
            label = f"{model_name} ({tracker_label})"
            options.append(label)
            mapping[label] = (model_key, tracker_key)

    return options, mapping


def run_single_job(
    *,
    model_type: str,
    tracker_key: str,
    video_name: str,
    video_path: Path,
    confidence: float,
    device: str,
    roi_data: dict[str, list[list[int]]],
) -> JobResult:
    output_video = (
        LABELED_VIDEOS_DIR / f"{model_type}_{tracker_key}_{video_name}.mp4"
    )
    output_metrics = (
        METRICS_DIR / f"{model_type}_{tracker_key}_{video_name}_stats.json"
    )

    roi = roi_data.get(video_path.name)

    try:
        from src.inference import get_detector
        from src.video_processor import process_video

        detector = get_detector(
            model_type,
            "package_sorting",
            device=device,
            confidence=confidence,
        )

        stats = process_video(
            video_path=str(video_path),
            detector=detector,
            output_path=str(output_video),
            model_type=model_type,
            tracker_key=cast(TrackerKey, tracker_key),
            confidence_threshold=confidence,
            show_progress=False,
            roi=roi,
        )

        save_metrics(stats, str(output_metrics), include_per_frame=True)
        summary = summarize_metrics(stats)

        return JobResult(
            model=model_type,
            tracker=tracker_key,
            video=video_name,
            success=True,
            output_video=output_video,
            output_metrics=output_metrics,
            summary=summary,
            error=None,
        )

    except Exception as e:
        return JobResult(
            model=model_type,
            tracker=tracker_key,
            video=video_name,
            success=False,
            output_video=None,
            output_metrics=None,
            summary="",
            error=str(e),
        )


def init_session_state() -> None:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "results" not in st.session_state:
        st.session_state.results = []
    if "current_job" not in st.session_state:
        st.session_state.current_job = ""
    if "progress" not in st.session_state:
        st.session_state.progress = 0.0


def main() -> None:
    init_session_state()

    st.title("Inference Runner")
    st.markdown("Run object detection inference on videos")

    videos = get_input_videos()
    roi_data = load_roi_data()
    selection_options, selection_map = build_selection_options()

    with st.sidebar:
        st.header("Configuration")

        mode = st.radio(
            "Mode",
            options=[
                "Custom Selection",
                "Run All Combinations",
            ],
            index=0,
        )

        st.divider()

        st.subheader("Videos")
        if not videos:
            st.warning("No videos found in input_videos/")
            selected_videos: list[str] = []
        else:
            selected_videos = st.multiselect(
                "Select videos:",
                options=list(videos.keys()),
                default=(list(videos.keys())[:1] if videos else []),
            )

        st.divider()

        if mode == "Custom Selection":
            st.subheader("Model + Tracker")
            selected_combos = st.multiselect(
                "Select combinations:",
                options=selection_options,
                default=(selection_options[:1] if selection_options else []),
            )
        else:
            selected_combos = selection_options
            combo_count = len(selected_combos)
            st.info(f"Will run all {combo_count} model+tracker combinations")

        st.divider()

        st.subheader("Parameters")

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

        device = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda"],
            index=0,
        )

    LABELED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str, Path]] = []
    for combo_label in selected_combos:
        model_type, tracker_key = selection_map[combo_label]
        for video_name in selected_videos:
            video_path = videos.get(video_name)
            if video_path:
                jobs.append(
                    (
                        model_type,
                        tracker_key,
                        video_name,
                        video_path,
                    )
                )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_disabled = st.session_state.running or len(jobs) == 0
        run_button = st.button(
            "Run Inference",
            disabled=run_disabled,
            type="primary",
            use_container_width=True,
        )

    with col2:
        if len(jobs) == 0:
            st.warning("Select at least one video and model+tracker")
        else:
            st.info(f"{len(jobs)} job(s) queued")

    if run_button and not st.session_state.running:
        st.session_state.running = True
        st.session_state.results = []
        st.session_state.progress = 0.0

    if st.session_state.running:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        total_jobs = len(jobs)
        results: list[JobResult] = []

        for idx, job in enumerate(jobs):
            tracker_label = TRACKER_DISPLAY.get(job[1]) or job[1]
            model_name = MODELS.get(job[0]) or job[0]
            status_text.markdown(
                f"**Processing:** {model_name} "
                f"({tracker_label}) "
                f"on {job[2]} ({idx + 1}/{total_jobs})"
            )

            result = run_single_job(
                model_type=job[0],
                tracker_key=job[1],
                video_name=job[2],
                video_path=job[3],
                confidence=confidence,
                device=device,
                roi_data=roi_data,
            )
            results.append(result)
            progress_bar.progress((idx + 1) / total_jobs)

        st.session_state.running = False
        st.session_state.results = results
        status_text.markdown("**Completed**")
        st.rerun()

    if st.session_state.results:
        st.divider()
        st.header("Results")

        success_count = sum(1 for r in st.session_state.results if r.success)
        fail_count = len(st.session_state.results) - success_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Jobs", len(st.session_state.results))
        col2.metric("Successful", success_count)
        col3.metric("Failed", fail_count)

        for result in st.session_state.results:
            result = cast(JobResult, result)
            tracker_label = TRACKER_DISPLAY.get(result.tracker, result.tracker)
            model_name = MODELS.get(result.model, result.model)
            title = f"{model_name} ({tracker_label}) - {result.video}"

            if result.success:
                with st.expander(f"{title}", expanded=False):
                    st.text(result.summary)
                    if result.output_video and result.output_video.exists():
                        st.video(str(result.output_video))
            else:
                with st.expander(f"{title} - FAILED", expanded=True):
                    st.error(result.error)

        if st.button("Clear Results"):
            st.session_state.results = []
            st.rerun()


main()
