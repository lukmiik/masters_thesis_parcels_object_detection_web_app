import json
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from src.paths import (
    INPUT_VIDEOS_DIR,
    LABELED_VIDEOS_DIR,
    METRICS_DIR,
)

MODELS: dict[str, dict[str, str]] = {
    "yolov11": {
        "name": "YOLOv11",
        "color": "#FF6B6B",
        "video_key": "yolov11",
        "csv_key": "yolov11",
    },
    "rf_detr": {
        "name": "RF-DETR",
        "color": "#4ECDC4",
        "video_key": "rf_detr",
        "csv_key": "rfdetr",
    },
    "faster_rcnn": {
        "name": "Faster R-CNN",
        "color": "#45B7D1",
        "video_key": "faster_rcnn",
        "csv_key": "fasterrcnn",
    },
}

CSV_TO_MODEL: dict[str, str] = {v["csv_key"]: k for k, v in MODELS.items()}

TRACKERS_BY_MODEL: dict[str, dict[str, str]] = {
    "yolov11": {
        "botsort": "BoT-SORT",
        "bytetrack": "ByteTrack",
    },
    "rf_detr": {
        "sort": "SORT",
        "bytetrack": "ByteTrack",
    },
    "faster_rcnn": {
        "sort": "SORT",
        "bytetrack": "ByteTrack",
    },
}

TRACKER_LABELS: dict[str, str] = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "sort": "SORT",
    "none": "None",
}


def get_input_videos() -> dict[str, str]:
    videos: dict[str, str] = {}
    if INPUT_VIDEOS_DIR.exists():
        for video_path in sorted(INPUT_VIDEOS_DIR.glob("*.mp4")):
            videos[video_path.stem] = video_path.stem.replace("_", " ").title()
    return videos


@st.cache_data
def load_comparison_summary() -> pd.DataFrame:
    csv_path = METRICS_DIR / "comparison_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_metrics_json(*, model: str, tracker: str, video_name: str) -> dict:
    json_path = METRICS_DIR / f"{model}_{tracker}_{video_name}_stats.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def get_video_path(
    *, model: str, tracker: str, video_name: str
) -> Path | None:
    video_key = MODELS[model]["video_key"]
    video_path = LABELED_VIDEOS_DIR / f"{video_key}_{tracker}_{video_name}.mp4"
    return video_path if video_path.exists() else None


def format_model_tracker_label(*, csv_key: str, tracker: str) -> str:
    model_key = CSV_TO_MODEL.get(csv_key, csv_key)
    model_name = MODELS.get(model_key, {}).get("name", csv_key)
    tracker_label = TRACKER_LABELS.get(tracker, tracker)
    return f"{model_name} ({tracker_label})"


def filter_summary(
    *,
    df: pd.DataFrame,
    video_name: str,
    selections: list[tuple[str, str]],
) -> pd.DataFrame:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        return video_df

    if "tracker" not in video_df.columns:
        video_df["tracker"] = "none"

    if not selections:
        return video_df

    allowed = [
        (MODELS[model]["csv_key"], tracker) for model, tracker in selections
    ]
    video_df["model_tracker"] = list(
        zip(
            video_df["model"],
            video_df["tracker"],
            strict=False,
        )
    )
    filtered = video_df[video_df["model_tracker"].isin(allowed)]
    return cast(pd.DataFrame, filtered)


def create_fps_chart(*, df: pd.DataFrame, video_name: str) -> None:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        st.warning("No FPS data available")
        return

    chart_data: list[dict[str, object]] = []
    for _, row in video_df.iterrows():
        csv_key = cast(str, row["model"])
        tracker_key = cast(str, row.get("tracker", "none"))
        chart_data.append(
            {
                "Model (Tracker)": format_model_tracker_label(
                    csv_key=csv_key,
                    tracker=tracker_key,
                ),
                "FPS": row["avg_inference_fps"],
            }
        )

    chart_df = pd.DataFrame(chart_data)
    st.bar_chart(chart_df.set_index("Model (Tracker)")["FPS"])


def create_inference_time_chart(*, df: pd.DataFrame, video_name: str) -> None:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        st.warning("No inference time data available")
        return

    chart_data: list[dict[str, object]] = []
    for _, row in video_df.iterrows():
        csv_key = cast(str, row["model"])
        tracker_key = cast(str, row.get("tracker", "none"))
        chart_data.append(
            {
                "Model (Tracker)": format_model_tracker_label(
                    csv_key=csv_key,
                    tracker=tracker_key,
                ),
                "Time (ms)": row["avg_inference_time_ms"],
            }
        )

    chart_df = pd.DataFrame(chart_data)
    st.bar_chart(chart_df.set_index("Model (Tracker)")["Time (ms)"])


def create_detection_density_chart(
    *, df: pd.DataFrame, video_name: str
) -> None:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        st.warning("No detection data available")
        return

    chart_data: list[dict[str, object]] = []
    for _, row in video_df.iterrows():
        csv_key = cast(str, row["model"])
        tracker_key = cast(str, row.get("tracker", "none"))
        chart_data.append(
            {
                "Model (Tracker)": format_model_tracker_label(
                    csv_key=csv_key,
                    tracker=tracker_key,
                ),
                "Detections/Frame": row["detection_density_per_frame"],
            }
        )

    chart_df = pd.DataFrame(chart_data)
    st.bar_chart(chart_df.set_index("Model (Tracker)")["Detections/Frame"])


def create_confidence_chart(*, df: pd.DataFrame, video_name: str) -> None:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        st.warning("No confidence data available")
        return

    chart_data: list[dict[str, object]] = []
    for _, row in video_df.iterrows():
        csv_key = cast(str, row["model"])
        tracker_key = cast(str, row.get("tracker", "none"))
        chart_data.append(
            {
                "Model (Tracker)": format_model_tracker_label(
                    csv_key=csv_key,
                    tracker=tracker_key,
                ),
                "Avg Confidence": row["avg_confidence"],
            }
        )

    chart_df = pd.DataFrame(chart_data)
    st.bar_chart(chart_df.set_index("Model (Tracker)")["Avg Confidence"])


def display_metrics_table(*, df: pd.DataFrame, video_name: str) -> None:
    video_df = cast(
        pd.DataFrame,
        df[df["video_file"] == f"{video_name}.mp4"].copy(),
    )
    if video_df.empty:
        st.warning("No metrics available")
        return

    class_columns = [c for c in video_df.columns if c in ("box", "mailer")]

    display_df = video_df[
        [
            "model",
            "tracker",
            "avg_inference_fps",
            "avg_inference_time_ms",
            "total_detections",
            "detection_density_per_frame",
            "avg_confidence",
            "min_confidence",
            "max_confidence",
        ]
        + class_columns
    ].copy()
    renamed = cast(pd.DataFrame, display_df).copy()
    col_names = [
        "Model",
        "Tracker",
        "FPS",
        "Inference (ms)",
        "Total Detections",
        "Detections/Frame",
        "Avg Conf",
        "Min Conf",
        "Max Conf",
    ]
    if class_columns:
        col_names.extend([c.title() for c in class_columns])
    renamed.columns = col_names
    renamed["Model (Tracker)"] = renamed.apply(
        lambda row: format_model_tracker_label(
            csv_key=cast(str, row["Model"]),
            tracker=cast(str, row.get("Tracker", "none")),
        ),
        axis=1,
    )
    final_df_cols: list[str] = [
        "Model (Tracker)",
        "FPS",
        "Inference (ms)",
        "Total Detections",
        "Detections/Frame",
        "Avg Conf",
        "Min Conf",
        "Max Conf",
    ]
    if class_columns:
        final_df_cols.extend([c.title() for c in class_columns])
    final_df = renamed[final_df_cols].round(2)

    st.dataframe(
        final_df.reset_index(drop=True),
        width="stretch",
    )


def main() -> None:
    st.markdown(
        """
        <style>
        .stVideo > video {
            border-radius: 8px;
        }
        .metric-card {
            background: linear-gradient(
                135deg, #667eea 0%, #764ba2 100%
            );
            padding: 1rem;
            border-radius: 10px;
            color: white;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("Object Detection Model Comparison")
    st.markdown(
        "Compare YOLOv11, RF-DETR, and Faster R-CNN across different videos"
    )

    summary_df = load_comparison_summary()

    input_videos = get_input_videos()

    with st.sidebar:
        st.header("Settings")

        st.subheader("Videos")
        if input_videos:
            selected_video = st.radio(
                "Select video:",
                options=list(input_videos.keys()),
                format_func=lambda x: input_videos[x],
                label_visibility="collapsed",
            )
        else:
            st.warning("No videos found in input_videos/")
            selected_video = None

        st.divider()

        st.subheader("Display Options")
        show_metrics_table = st.checkbox("Show metrics table", value=True)
        show_charts = st.checkbox("Show comparison charts", value=True)

    if not selected_video:
        st.warning("Please add videos to the input_videos/ directory.")
        return

    st.subheader("Model + Tracker Selection")
    selection_options: list[str] = []
    selection_map: dict[str, tuple[str, str]] = {}
    for model_key in MODELS:
        trackers = TRACKERS_BY_MODEL.get(model_key, {})
        for tracker_key, tracker_label in trackers.items():
            label = f"{MODELS[model_key]['name']} ({tracker_label})"
            selection_options.append(label)
            selection_map[label] = (model_key, tracker_key)

    default_selections: list[str] = []
    for model_key, trackers in TRACKERS_BY_MODEL.items():
        if trackers:
            tracker_key = next(iter(trackers))
            tracker_label = trackers[tracker_key]
            default_selections.append(
                f"{MODELS[model_key]['name']} ({tracker_label})"
            )

    selected_labels = st.multiselect(
        "Select up to 4 model+tracker pairs:",
        options=selection_options,
        default=default_selections,
        max_selections=4,
    )

    selections = [selection_map[label] for label in selected_labels]
    if not selections:
        st.warning("Please select at least one model and tracker.")
        return

    st.header(f"Video Comparison - {input_videos[selected_video]}")

    max_selections = selections[:4]
    rows = (len(max_selections) + 1) // 2
    for row_idx in range(rows):
        cols = st.columns(2)
        for col_idx in range(2):
            selection_idx = row_idx * 2 + col_idx
            if selection_idx >= len(max_selections):
                break
            model_key, tracker_key = max_selections[selection_idx]
            tracker_label = TRACKER_LABELS.get(tracker_key, tracker_key)
            with cols[col_idx]:
                st.subheader(f"{MODELS[model_key]['name']} ({tracker_label})")
                video_path = get_video_path(
                    model=model_key,
                    tracker=tracker_key,
                    video_name=selected_video,
                )

                if video_path:
                    st.video(str(video_path))
                else:
                    st.error(
                        "Video not found: "
                        f"{model_key}_{tracker_key}"
                        f"_{selected_video}.mp4"
                    )

    st.divider()

    filtered_summary = filter_summary(
        df=summary_df,
        video_name=selected_video,
        selections=max_selections,
    )

    if not filtered_summary.empty and (show_metrics_table or show_charts):
        st.header("Performance Metrics")

        if show_metrics_table:
            display_metrics_table(
                df=filtered_summary,
                video_name=selected_video,
            )

        if show_charts:
            chart_cols = st.columns(4)

            with chart_cols[0]:
                st.subheader("Inference FPS")
                create_fps_chart(
                    df=filtered_summary,
                    video_name=selected_video,
                )

            with chart_cols[1]:
                st.subheader("Inference Time")
                create_inference_time_chart(
                    df=filtered_summary,
                    video_name=selected_video,
                )

            with chart_cols[2]:
                st.subheader("Detection Density")
                create_detection_density_chart(
                    df=filtered_summary,
                    video_name=selected_video,
                )

            with chart_cols[3]:
                st.subheader("Avg Confidence")
                create_confidence_chart(
                    df=filtered_summary,
                    video_name=selected_video,
                )

    st.divider()
    st.caption(
        "Object Detection Model Comparison | "
        "Models: YOLOv11, RF-DETR, Faster R-CNN"
    )


main()
