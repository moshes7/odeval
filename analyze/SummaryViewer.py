import streamlit as st
import cv2
import numpy as np
import analyze.analyzeViewer2
import pandas
import analyze.FrameViewer
import analyze.DataVisualizer


def run_summary_viewer():
    tables = st.columns(2)
    with tables[0].expander("Confusion Matrix"):
        st.pyplot(st.session_state.analyzeViewer.get_total_plot_cm())
    with st.expander("Class"):
        st.session_state.analyzeViewer.analyzer.metrics_tables["class"][
            st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(include=['number']).columns] = \
            st.session_state.analyzeViewer.analyzer.metrics_tables["class"][
                st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(
                    include=['number']).columns].astype(float).round(3)
        st.dataframe(st.session_state.analyzeViewer.analyzer.metrics_tables["class"].style.applymap(
            analyze.DataVisualizer.text_highlighter))
    with tables[1].expander("Global Data"):
        st.table(st.session_state.analyzeViewer.analyzer.metrics_tables["global"])
