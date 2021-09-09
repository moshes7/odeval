import streamlit as st
import cv2
import numpy as np
import analyze.analyzeViewer2
import pandas
import analyze.FrameViewer
import analyze.SummaryViewer


def run(analyzer):
    if 'analyzeViewer' not in st.session_state:
        st.set_page_config(layout='wide')
        st.session_state.analyzeViewer = analyze.analyzeViewer2.AnalyzeViewer2(analyzer)
    app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
    if app_mode == "Summary":
        analyze.SummaryViewer.run_summary_viewer()

    elif app_mode == "Frame Viewer":
        analyze.FrameViewer.show_frame_viewer(len(analyzer.data))
