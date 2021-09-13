import streamlit as st

import analyze.FrameViewer
import analyze.SummaryViewer
import analyze.analyzeViewer2


def run(analyzer):
    if 'analyzeViewer' not in st.session_state:
        st.set_page_config(layout='wide')
        st.session_state.analyzeViewer = analyze.analyzeViewer2.AnalyzeViewer2(analyzer)
    app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
    st.session_state.analyzeViewer.analyzer.score_th = st.sidebar.slider("Score th:", 0.0, 1.0, st.session_state.analyzeViewer.analyzer.score_th)
    st.session_state.analyzeViewer.analyzer.iou_th = st.sidebar.slider("IOU th:", 0.0, 1.0, st.session_state.analyzeViewer.analyzer.iou_th)
    st.session_state.analyzeViewer.analyzer.bbox_match_method = st.sidebar.radio("Match Method:", ["iou", "pred_bbox_center"])
    if st.sidebar.button('Reset to default'):
        st.session_state.analyzeViewer.analyzer.score_th = 0.3
        st.session_state.analyzeViewer.analyzer.iou_th = 0.5
        st.session_state.analyzeViewer.analyzer.bbox_match_method = "iou"
    if app_mode == "Summary":
        analyze.SummaryViewer.run_summary_viewer()
    elif app_mode == "Frame Viewer":
        analyze.FrameViewer.show_frame_viewer(len(analyzer.data))

