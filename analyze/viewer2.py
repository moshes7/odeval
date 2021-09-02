import streamlit as st
import cv2
import numpy as np

if 'count' not in st.session_state:
    st.session_state.count = 0


def run(analyzer):
    st.title("Test")
    app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
    if app_mode == "Summary":
        st.write("reach")
    elif app_mode == "Frame Viewer":

        x = st.columns(1)

        b1, b2, b3 = st.columns(3)

        b1.button("Back", on_click=click_minus)
        b3.button("Forward", on_click=click_plus)
        if st.session_state.count < 0:
            st.session_state.count = len(analyzer.data) - 1
        elif st.session_state.count > len(analyzer.data) - 1:
            st.session_state.count = 0
        b2.write(st.session_state.count)
        x[0].image(load_image(analyzer.data[st.session_state.count]["image_path"]).astype(np.uint8),
                 use_column_width=True)

@st.cache(show_spinner=False)
def load_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def click_plus():
    st.session_state.count += 1


def click_minus():
    st.session_state.count -= 1
