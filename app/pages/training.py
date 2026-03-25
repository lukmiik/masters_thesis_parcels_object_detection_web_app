import subprocess
from pathlib import Path

import streamlit as st

from src.paths import TRAINING_NOTEBOOKS_DIR, TRAINING_OUTPUTS_DIR

NOTEBOOKS: dict[str, dict[str, str | int]] = {
    "YOLOv11": {
        "file": "train_yolov11.ipynb",
        "default_epochs": 3,
        "description": "Train YOLOv11 object detection model",
    },
    "RF-DETR": {
        "file": "train_rf_detr.ipynb",
        "default_epochs": 2,
        "description": "Train RF-DETR transformer-based detector",
    },
    "Faster R-CNN": {
        "file": "train_faster_rcnn.ipynb",
        "default_epochs": 2,
        "description": "Train Faster R-CNN with MobileNetV3 backbone",
    },
}


def run_notebook(*, notebook_path: Path, epochs: int) -> tuple[bool, str]:
    """Run a notebook with papermill, passing epochs parameter."""
    TRAINING_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_OUTPUTS_DIR / f"{notebook_path.stem}_output.ipynb"

    cmd = [
        "uv",
        "run",
        "papermill",
        str(notebook_path),
        str(output_path),
        "-p",
        "EPOCHS",
        str(epochs),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=notebook_path.parent,
            check=False,
        )
        if result.returncode == 0:
            return True, f"Completed successfully. Output: {output_path}"
        return False, f"Error:\n{result.stderr}"
    except Exception as e:
        return False, f"Exception: {e}"


def main() -> None:
    st.title("Training Notebooks Runner")

    st.markdown("Select notebooks to train and optionally customize epochs.")

    st.subheader("Select Notebooks")

    selected_notebooks: dict[str, bool] = {}
    custom_epochs: dict[str, int] = {}

    for name, config in NOTEBOOKS.items():
        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            selected_notebooks[name] = st.checkbox(name, key=f"cb_{name}")

        with col2:
            st.caption(config["description"])

        with col3:
            custom_epochs[name] = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=config["default_epochs"],
                key=f"epochs_{name}",
            )

    st.divider()

    col_run, col_all = st.columns(2)

    with col_run:
        run_selected = st.button("Run Selected", type="primary")

    with col_all:
        run_all = st.button("Run All Notebooks")

    if run_all:
        for name in NOTEBOOKS:
            selected_notebooks[name] = True

    if run_selected or run_all:
        notebooks_to_run = [
            name for name, selected in selected_notebooks.items() if selected
        ]

        if not notebooks_to_run:
            st.warning("No notebooks selected.")
            return

        st.subheader("Training Progress")

        for name in notebooks_to_run:
            config = NOTEBOOKS[name]
            notebook_path = TRAINING_NOTEBOOKS_DIR / str(config["file"])
            epochs = custom_epochs[name]

            with st.spinner(f"Running {name}..."):
                st.write(f"Training with {epochs} epochs")

                success, message = run_notebook(
                    notebook_path=notebook_path,
                    epochs=epochs,
                )

                if success:
                    st.success(f"{name}: {message}")
                else:
                    st.error(f"{name}: {message}")


main()
