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
        # st.image(load_image(analyzer.data[st.session_state.count]["image_path"]).astype(np.uint8),
        #          use_column_width=True)
        # b1, _, b2 = st.columns(3)
        # b1.button("Back", on_click=click)
        # b2.button("Forward", on_click=click)
        st.write("reach2")



# class Viewer:
#     def __init__(self, analyzer):
#         st.title("Test")
#         self.analyzer = analyzer
#         self.app_mode = st.sidebar.selectbox("Mode", ["Summary", "Frame Viewer", "EXIT"])
#         self.current_id: int = 0
#         self.last_id = 1
#
#     def run(self) -> None:
#         while 1:
#             if self.app_mode == "Summary":
#                 if self.last_id != self.current_id:
#                     st.write("testy")
#             elif self.app_mode == "Frame Viewer":
#                 if self.last_id != self.current_id:
#                     self.init_page_image()
#             elif self.app_mode == "EXIT":
#                 pass
#

@st.cache(show_spinner=False)
def load_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def click():
    st.session_state.count += 1
