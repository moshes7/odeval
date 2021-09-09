import pandas
import streamlit as st
from analyze.analyzeViewer2 import ConfusionMatrix
import analyze.DataVisualizer
if 'count' not in st.session_state:
    st.session_state.count = 0


def show_frame_viewer(num_of_photos: int) -> None:
    is_cm_available: str = st.sidebar.columns(2)[0].select_slider("Enable Confusion Matrix", ["YES", "NO"], value="NO")
    l: list = st.columns(3)
    l2: list = st.columns(12)
    l3: list = st.columns(3)
    insert_photos_buttons(l2, num_of_photos)
    st.session_state.count = l3[0].slider("Choose a frame", 0, num_of_photos, st.session_state.count)
    l[0].image(st.session_state.analyzeViewer.get_drawn_image(st.session_state.count), width=700)
    show_analysis(is_cm_available, l)


def click_plus() -> None:
    st.session_state.count += 1


def click_minus() -> None:
    st.session_state.count -= 1


def get_frame_analysis(frame_id) -> ConfusionMatrix:
    st.session_state.analyzeViewer.frame_id = frame_id
    return st.session_state.analyzeViewer.get_confusion_metrix_for_frame()


def insert_photos_buttons(buttons, num_of_photos) -> None:
    buttons[0].button("Back", on_click=click_minus)
    buttons[4].button("Forward", on_click=click_plus)
    if st.session_state.count < 0:
        st.session_state.count = num_of_photos - 1
    elif st.session_state.count > num_of_photos - 1:
        st.session_state.count = 0


def show_analysis(is_cm_available, page_part) -> None:
    cm = get_frame_analysis(st.session_state.count)
    page_part[2].table(pandas.DataFrame.from_dict(cm.metrics_tables["global"]))
    with st.expander("Class"):
        st.dataframe(cm.metrics_tables["class"].style.applymap(analyze.DataVisualizer.text_highlighter),height=1000)
    if is_cm_available == "YES":
        with st.expander("Confusion Matrix"):
            st.pyplot(st.session_state.analyzeViewer.get_plot_cm(cm)[1])
