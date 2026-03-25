import streamlit as st

view_page = st.Page("pages/view.py", title="Model Comparison", default=True)
inference_page = st.Page("pages/inference.py", title="Inference Runner")
roi_page = st.Page("pages/roi_editor.py", title="ROI Editor")
training_page = st.Page("pages/training.py", title="Training")

pg = st.navigation([view_page, inference_page, roi_page, training_page])

st.set_page_config(
    page_title="Object Detection Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg.run()
