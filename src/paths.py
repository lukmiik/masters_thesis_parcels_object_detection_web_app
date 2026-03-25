from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

INPUT_VIDEOS_DIR: Path = PROJECT_ROOT / "input_videos"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LABELED_VIDEOS_DIR: Path = OUTPUTS_DIR / "labeled_videos"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
ROI_FILE: Path = PROJECT_ROOT / "roi.json"
TRAINING_NOTEBOOKS_DIR: Path = PROJECT_ROOT / "training_notebooks"
TRAINING_OUTPUTS_DIR: Path = TRAINING_NOTEBOOKS_DIR / "outputs"
