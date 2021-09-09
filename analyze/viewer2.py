import streamlit as st
import cv2
import numpy as np
import analyze.analyzeViewer2
import pandas

if 'count' not in st.session_state:
    st.session_state.count = 0


def run(analyzer):
    if 'analyzeViewer' not in st.session_state:
        st.set_page_config(layout='wide')
        st.session_state.analyzeViewer = analyze.analyzeViewer2.AnalyzeViewer2(analyzer)
    app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
    if app_mode == "Summary":

        tables = st.columns(2)
        with tables[0].expander("Confusion Matrix"):
            st.pyplot(st.session_state.analyzeViewer.get_total_plot_cm())
        with st.expander("Class"):
            st.session_state.analyzeViewer.analyzer.metrics_tables["class"][st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(include=['number']).columns] = st.session_state.analyzeViewer.analyzer.metrics_tables["class"][st.session_state.analyzeViewer.analyzer.metrics_tables["class"].select_dtypes(include=['number']).columns].astype(float).round(3)
            st.dataframe(st.session_state.analyzeViewer.analyzer.metrics_tables["class"].style.applymap(lambda v: 'color:red;' if (type(v) != str and v > 0) else None))
        with tables[1].expander("Global Data"):
            st.table(st.session_state.analyzeViewer.analyzer.metrics_tables["global"])

    elif app_mode == "Frame Viewer":
        is_cm_available = st.sidebar.columns(2)[0].select_slider("Enable Confusion Matrix", ["YES", "NO"], value="NO")
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
        with st.expander("Class"):
            st.table(pandas.DataFrame.from_dict(cm.metrics_tables["class"]))
        if is_cm_available == "YES":
            with st.expander("Confusion Matrix"):
                st.pyplot(st.session_state.analyzeViewer.get_plot_cm(cm)[1])


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


