import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.paths import INPUT_VIDEOS_DIR, ROI_FILE


def get_video_files() -> list[str]:
    if not INPUT_VIDEOS_DIR.exists():
        return []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [
        f.name
        for f in INPUT_VIDEOS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]


def get_video_properties(*, video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frame_count, width, height


def get_frame(*, video_path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_number}")

    return frame


def load_roi_config(*, video_name: str) -> dict[str, Any] | None:
    if not ROI_FILE.exists():
        return None

    if ROI_FILE.stat().st_size == 0:
        return None

    with open(ROI_FILE, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    video_data = data.get(video_name)
    if video_data is None:
        return None

    if not isinstance(video_data, dict):
        return None

    if "roi" not in video_data and "canvas" not in video_data:
        return None

    return video_data


def save_roi_config(
    *,
    video_name: str,
    roi_points: list[list[int]],
    canvas_data: dict[str, Any] | None,
) -> None:
    data: dict[str, Any] = {}
    if ROI_FILE.exists() and ROI_FILE.stat().st_size > 0:
        with open(ROI_FILE, encoding="utf-8") as f:
            data = json.load(f)

    data[video_name] = {"roi": roi_points}
    if canvas_data is not None:
        data[video_name]["canvas"] = canvas_data

    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def convert_canvas_result_to_roi(
    *,
    canvas_result: Any,
    scale_factor: float = 1.0,
) -> list[list[int]] | None:
    if canvas_result is None:
        return None

    json_data = getattr(canvas_result, "json_data", None)
    if json_data is None:
        return None

    objects = json_data.get("objects", [])
    if not objects:
        return None

    for obj in objects:
        obj_type = obj.get("type", "")
        if obj_type in ("polygon", "path"):
            points = obj.get("path", [])
            if not points:
                continue

            roi_points: list[list[int]] = []
            for point in points:
                if isinstance(point, list) and len(point) >= 3:
                    cmd = point[0]
                    if cmd in ("M", "L"):
                        x = int(point[1] / scale_factor)
                        y = int(point[2] / scale_factor)
                        roi_points.append([x, y])

            if len(roi_points) >= 3:
                return roi_points

    return None


def _needs_explicit_close(*, path: list[list[Any]]) -> bool:
    if len(path) < 2:
        return False

    last_cmd = path[-1]
    if not (
        isinstance(last_cmd, list) and len(last_cmd) > 0 and last_cmd[0] == "z"
    ):
        return False

    first_point = _find_first_point(path=path)
    if first_point is None:
        return False

    second_last = path[-2]
    if not (isinstance(second_last, list) and len(second_last) >= 3):
        return True

    if second_last[0] != "L":
        return True

    # Check whether the last L actually returns to the first point
    return (
        float(second_last[1]) != first_point[0]
        or float(second_last[2]) != first_point[1]
    )


def _find_first_point(*, path: list[list[Any]]) -> list[float] | None:
    for cmd in path:
        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] == "M":
            return [float(cmd[1]), float(cmd[2])]
    return None


def _add_explicit_close(*, path: list[list[Any]]) -> list[list[Any]]:
    first_point = _find_first_point(path=path)
    if first_point is None:
        return path

    new_path = path[:-1]
    new_path.append(["L", first_point[0], first_point[1]])
    new_path.append(["z"])
    return new_path


def _is_polygon_closed(*, canvas_data: dict[str, Any]) -> bool:
    """Check if the polygon has a closing 'z' command."""
    if not isinstance(canvas_data, dict):
        return False

    objects = canvas_data.get("objects", [])
    if not objects:
        return False

    for obj in objects:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            if path:
                last_cmd = path[-1]
                if (
                    isinstance(last_cmd, list)
                    and len(last_cmd) > 0
                    and last_cmd[0] == "z"
                ):
                    return True

    return False


def ensure_explicit_polygon_close(
    *,
    canvas_data: dict[str, Any],
) -> dict[str, Any]:
    """Ensure polygon paths have explicit closing lines."""
    if not isinstance(canvas_data, dict):
        return canvas_data

    objects = canvas_data.get("objects", [])
    if not objects:
        return canvas_data

    modified_objects = []
    for obj in objects:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            if path and len(path) >= 3 and _needs_explicit_close(path=path):
                path = _add_explicit_close(path=path)
                obj = {**obj, "path": path}
        modified_objects.append(obj)

    return {**canvas_data, "objects": modified_objects}


def create_initial_drawing(
    *,
    roi_points: list[list[int]] | None,
    scale_factor: float = 1.0,
) -> dict[str, Any] | None:
    if roi_points is None or len(roi_points) < 3:
        return None

    path_data: list[list[str | float]] = []
    for i, point in enumerate(roi_points):
        x = float(point[0]) * scale_factor
        y = float(point[1]) * scale_factor
        if i == 0:
            path_data.append(["M", x, y])
        else:
            path_data.append(["L", x, y])

    first_x = float(roi_points[0][0]) * scale_factor
    first_y = float(roi_points[0][1]) * scale_factor
    path_data.append(["L", first_x, first_y])
    path_data.append(["z"])

    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "path",
                "version": "4.4.0",
                "originX": "center",
                "originY": "center",
                "left": 0,
                "top": 0,
                "pathOffset": {"x": 0.0, "y": 0.0},
                "fill": "rgba(255, 0, 0, 0.3)",
                "stroke": "red",
                "strokeWidth": 2,
                "path": path_data,
            },
        ],
    }


def main() -> None:
    st.title("ROI Mask Editor")
    st.markdown("Draw polygon masks on video frames for object detection.")

    video_files = get_video_files()
    if not video_files:
        st.warning("No video files found in 'input_videos' folder.")
        return

    selected_video = st.selectbox("Select Video", video_files)
    if selected_video is None:
        return

    video_path = INPUT_VIDEOS_DIR / selected_video

    try:
        frame_count, width, height = get_video_properties(
            video_path=video_path,
        )
    except ValueError as e:
        st.error(f"Error loading video: {e}")
        return

    st.markdown(
        f"**Resolution:** {width}x{height} | **Frames:** {frame_count}"
    )

    frame_number = st.slider(
        "Select Frame",
        min_value=0,
        max_value=frame_count - 1,
        value=0,
        step=1,
    )

    try:
        frame = get_frame(video_path=video_path, frame_number=frame_number)
    except ValueError as e:
        st.error(f"Error reading frame: {e}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    max_canvas_width = 800
    scale_factor = min(1.0, max_canvas_width / width)
    canvas_width = int(width * scale_factor)
    canvas_height = int(height * scale_factor)

    roi_config = load_roi_config(video_name=selected_video)
    existing_roi = None
    saved_canvas = None
    if roi_config is not None:
        existing_roi = roi_config.get("roi")
        saved_canvas = roi_config.get("canvas")

    session_canvas_key = f"roi_canvas_{selected_video}"
    reset_triggered_key = f"roi_canvas_reset_triggered_{selected_video}"
    reset_was_triggered = st.session_state.get(reset_triggered_key, False)

    session_canvas = st.session_state.get(session_canvas_key)

    if reset_was_triggered:
        st.session_state.pop(reset_triggered_key, None)
        drawing_source = create_initial_drawing(
            roi_points=existing_roi,
            scale_factor=scale_factor,
        )
    elif isinstance(session_canvas, dict):
        drawing_source = copy.deepcopy(session_canvas)
    elif isinstance(saved_canvas, dict):
        drawing_source = copy.deepcopy(saved_canvas)
    else:
        drawing_source = create_initial_drawing(
            roi_points=existing_roi,
            scale_factor=scale_factor,
        )

    st.markdown("### Draw Polygon ROI")
    st.markdown(
        "Use the polygon tool to draw a mask. "
        "Click points to create the polygon shape."
    )

    pil_image_resized = pil_image.resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS,
    )

    initial_drawing = drawing_source

    reset_token_key = f"roi_canvas_reset_{selected_video}"
    reset_token = int(st.session_state.get(reset_token_key, 0))
    canvas_key = f"canvas_{selected_video}_{reset_token}"

    max_width = max_canvas_width
    style = f"#canvas_container_{{max-width: {max_width}px; margin: 0 auto;}}"
    st.markdown(
        f"<style>{style}</style>",
        unsafe_allow_html=True,
    )

    canvas_kwargs: dict[str, Any] = {
        "fill_color": "rgba(255, 0, 0, 0.3)",
        "stroke_width": 2,
        "stroke_color": "red",
        "background_image": pil_image_resized,
        "height": canvas_height,
        "width": canvas_width,
        "drawing_mode": "polygon",
        "key": canvas_key,
        "display_toolbar": False,
    }
    if initial_drawing is not None:
        canvas_kwargs["initial_drawing"] = initial_drawing

    with st.container():
        st.markdown(
            '<div id="canvas_container">',
            unsafe_allow_html=True,
        )
        canvas_result = st_canvas(**canvas_kwargs)
        st.markdown("</div>", unsafe_allow_html=True)

    reset_triggered_key = f"roi_canvas_reset_triggered_{selected_video}"

    pre_clear_canvas_data: dict[str, Any] | None = st.session_state.get(
        session_canvas_key,
    )

    if canvas_result is not None and canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            processed_data = ensure_explicit_polygon_close(
                canvas_data=canvas_result.json_data,
            )
            # Only update session state if polygon is closed (has 'z' command)
            # This prevents unnecessary re-renders during drawing
            is_polygon_closed = _is_polygon_closed(canvas_data=processed_data)
            current_session_data = st.session_state.get(session_canvas_key)
            data_changed = current_session_data != processed_data

            if is_polygon_closed and data_changed:
                st.session_state[session_canvas_key] = processed_data
                pre_clear_canvas_data = processed_data
                st.session_state.pop(reset_triggered_key, None)
        else:
            had_previous_drawing = (
                st.session_state.get(session_canvas_key) is not None
            )
            if had_previous_drawing:
                st.session_state[reset_triggered_key] = True
                st.session_state[reset_token_key] = reset_token + 1
            st.session_state.pop(session_canvas_key, None)

    col_save, col_clear, _spacer = st.columns([2, 1, 5])

    with col_save:
        if st.button("Save ROI Configuration", type="primary"):
            save_canvas_data = pre_clear_canvas_data

            class _CanvasProxy:
                def __init__(self, *, json_data: dict[str, Any]) -> None:
                    self.json_data = json_data

            roi_points = None
            if save_canvas_data is not None:
                roi_points = convert_canvas_result_to_roi(
                    canvas_result=_CanvasProxy(json_data=save_canvas_data),
                    scale_factor=scale_factor,
                )

            if roi_points is not None and len(roi_points) >= 3:
                save_roi_config(
                    video_name=selected_video,
                    roi_points=roi_points,
                    canvas_data=save_canvas_data,
                )
                st.success(f"ROI saved for '{selected_video}'")
            else:
                st.error("Please draw a valid polygon (at least 3 points)")

    with col_clear:
        if st.button("Clear ROI"):
            file_exists = ROI_FILE.exists() and ROI_FILE.stat().st_size > 0
            if file_exists:
                with open(ROI_FILE, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                if selected_video in data:
                    del data[selected_video]
                    with open(ROI_FILE, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            st.session_state.pop(session_canvas_key, None)
            st.session_state.pop(reset_triggered_key, None)
            st.session_state[reset_token_key] = reset_token + 1
            st.success(f"ROI cleared for '{selected_video}'")
            st.rerun()


main()
