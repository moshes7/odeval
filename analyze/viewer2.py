import streamlit as st
import cv2
import numpy as np
import analyze.analyzeViewer2
import pandas

if 'count' not in st.session_state:
    st.session_state.count = 1


def run(analyzer):
    if 'analyzeViewer' not in st.session_state:
        st.set_page_config(layout='wide')
        st.session_state.analyzeViewer = analyze.analyzeViewer2.AnalyzeViewer2(analyzer)
    app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
    if app_mode == "Summary":
        st.write("reach")
    elif app_mode == "Frame Viewer":

        x = st.columns(3)

        b = st.columns(12)
        a = st.columns(3)
        b[0].button("Back", on_click=click_minus)
        b[4].button("Forward", on_click=click_plus)
        if st.session_state.count < 0:
            st.session_state.count = len(analyzer.data) - 1
        elif st.session_state.count > len(analyzer.data) - 1:
            st.session_state.count = 0
        st.session_state.count = a[0].slider("Choose a frame", 0, len(analyzer.data), st.session_state.count)
        b[2].write(st.session_state.count)
        x[0].image(st.session_state.analyzeViewer.get_drawn_image(st.session_state.count), width=700)
        cm = get_frame_analysis(st.session_state.count)
        x[2].table(pandas.DataFrame.from_dict(cm.metrics_tables["global"]))
        with st.expander("Confusion Metrix"):
            st.pyplot(st.session_state.analyzeViewer.get_plot_cm(cm)[1])
        with st.expander("Class"):
            st.table(pandas.DataFrame.from_dict(cm.metrics_tables["class"]))

        # rerun


@st.cache(show_spinner=False)
def load_image(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def click_plus():
    st.session_state.count += 1


def click_minus():
    st.session_state.count -= 1


def get_frame_analysis(frame_id):
    st.session_state.analyzeViewer.frame_id = frame_id
    cm = st.session_state.analyzeViewer.get_confusion_metrix_for_frame()
    return cm
    # x = pandas.DataFrame.from_dict(cm.metrics["global"])
    # # #st.dataframe(data=x, width=500, height=500)
    # x1, x2, _ = st.columns(3)
    # x1.table(x)
    # # _.table(cm.metrics["class"])
    # pass
