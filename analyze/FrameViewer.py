import pandas
import streamlit as st
import analyze.DataVisualizer

if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.micro_or_macro = "macro"


def show_frame_viewer(num_of_photos: int) -> None:
    if 'frames_dict' not in st.session_state:
        st.session_state.frames_dict = {}
        for x in range(num_of_photos):
            st.session_state.frames_dict[x] = x
    is_cm_available: str = st.sidebar.radio("Enable Confusion Matrix", ["NO", "YES"])
    sort_key: str = st.sidebar.radio("Sort by:", ["Nothing", "Recall", "Precision", "FDR"])
    st.session_state.micro_or_macro = st.sidebar.radio("Choose graph type:", ["macro", "micro"])
    l: list = st.columns(3)
    l2: list = st.columns(12)
    l3: list = st.columns(3)
    insert_photos_buttons(l2, num_of_photos)
    st.session_state.count = l3[0].slider("Choose a frame", 0, num_of_photos - 1, st.session_state.count)
    graph_data = st.session_state.analyzeViewer.get_all_frames(num_of_photos)
    graph_data = sort_images(graph_data, num_of_photos, sort_key)
    show_analysis(is_cm_available, l)
    l[0].image(st.session_state.analyzeViewer.get_drawn_image(st.session_state.frames_dict[st.session_state.count]),
               width=700)
    organized_data = separate_graph_list(graph_data)
    show_charts(organized_data, sort_key)


def insert_photos_buttons(buttons, num_of_photos) -> None:
    if buttons[0].button("Back"):
        st.session_state.count -= 1
    if buttons[4].button("Forward"):
        st.session_state.count += 1
    if st.session_state.count < 0:
        st.session_state.count = num_of_photos - 1
    elif st.session_state.count > num_of_photos - 1:
        st.session_state.count = 0


def show_analysis(is_cm_available, page_part) -> None:
    cm = st.session_state.analyzeViewer.get_confusion_matrix_for_frame(
        st.session_state.frames_dict[st.session_state.count])
    page_part[2].table(pandas.DataFrame.from_dict(cm.metrics_tables["global"]))
    with st.expander("Class"):
        st.dataframe(cm.metrics_tables["class"].style.applymap(analyze.DataVisualizer.text_highlighter), height=1000)
    if is_cm_available == "YES":
        with st.expander("Confusion Matrix"):
            st.pyplot(st.session_state.analyzeViewer.get_plot_cm(cm)[1])


def separate_graph_list(data):
    recall = []
    pre = []
    fdr = []
    for d in data:
        # data_without_numbers.append([d[0], d[1], d[2]])
        recall.append(d[0])
        pre.append(d[1])
        fdr.append(d[2])
    return [recall, pre, fdr]  # , data_without_numbers]


def sort_images(data, num_of_photos, key):
    if key == "Nothing":
        key = lambda tmp: tmp[3]
    elif key == "Recall":
        key = lambda tmp: tmp[0]
    elif key == "Precision":
        key = lambda tmp: tmp[1]
    else:
        key = lambda tmp: tmp[2]
    sorted_data = sorted(data, key=key)
    for x in range(num_of_photos):
        st.session_state.frames_dict[x] = sorted_data[x][3]
    return sorted_data


def show_charts(data, key):
    if key == "Recall":
        with st.expander("Recall"):
            st.line_chart(pandas.DataFrame(data[0]))
    elif key == "Precision":
        with st.expander("Precision"):
            st.line_chart(pandas.DataFrame(data[1]))
    elif key == "FDR":
        with st.expander("FDR"):
            st.line_chart(pandas.DataFrame(data[2]))
    else:
        with st.expander("Recall"):
            st.line_chart(pandas.DataFrame(data[0]))
        with st.expander("Precision"):
            st.line_chart(pandas.DataFrame(data[1]))
        with st.expander("FDR"):
            st.line_chart(pandas.DataFrame(data[2]))
    # with st.expander("Mixed"):
    #     st.line_chart(pandas.DataFrame(data[3], columns=["Recall", "Precision", "FDR"]))
