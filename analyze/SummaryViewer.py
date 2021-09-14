import streamlit as st

import analyze.DataVisualizer
import analyze.FrameViewer
import analyze.analyzeViewer2


def run_summary_viewer():
    tables = st.columns(2)
    tables[0].pyplot(st.session_state.analyzeViewer.get_total_plot_cm())
    st.session_state.analyzeViewer.analyzer.metrics_tables["class"][
        st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(include=['number']).columns] = \
        st.session_state.analyzeViewer.analyzer.metrics_tables["class"][
            st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(
                include=['number']).columns].astype(float).round(3)
    st.dataframe(st.session_state.analyzeViewer.analyzer.metrics_tables["class"].style.applymap(
        analyze.DataVisualizer.text_highlighter))
    tables[1].table(st.session_state.analyzeViewer.analyzer.metrics_tables["global"])
