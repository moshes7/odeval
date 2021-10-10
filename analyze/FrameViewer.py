import pandas
import streamlit as st
import analyze.DataVisualizer
import time

DELAY_BETWEEN_PICS = 0.15


def show_frame_viewer(num_of_photos: int) -> None:
    if 'frames_dict' not in st.session_state:  # Initializes all variables.
        st.session_state.frames_dict = {}
        st.session_state.reverse = True
        st.session_state.steps = 1
        st.session_state.key_counter = 0
        st.session_state.play = False
        st.session_state.prev_sort_key = ""
        st.session_state.organized_data = ""
        st.session_state.num_of_photos = num_of_photos
        for x in range(num_of_photos):  # Initializes the frames dict, which can change the order of the frames.
            st.session_state.frames_dict[x] = x

    is_cm_available: str = st.sidebar.radio("Enable Confusion Matrix",
                                            ["NO", "YES"])  # Whether or not a confusion matrix will be shown.
    sort_key: str = st.sidebar.radio("Sort by:", ["Nothing", "Recall", "Precision", "FDR", "Miss Detection Rate",
                                                  "False Detection Rate Vs GT", "False Detection Rate Vs Pred"])
    st.session_state.reverse = True if st.sidebar.radio("Ascending/Descending",
                                                        ["Ascending",
                                                         "Descending"]) == "Descending" else False  # Whether or not reverse the graph.
    if sort_key == "Nothing":  # If you don't want the frames in a specific order, you will be able to create a new graph that will contain costumize data.
        sort_key = st.sidebar.multiselect("Show on graph:",
                                          ["Recall", "Precision", "FDR", "Miss Detection Rate",
                                           "False Detection Rate Vs GT", "False Detection Rate Vs Pred"])
    st.session_state.micro_or_macro = st.sidebar.radio("Choose graph type:", ["macro", "micro"])
    l: list = st.columns(3)
    l2: list = st.columns(6)
    l3: list = st.columns(3)

    if l2[4].button("Play"):
        st.session_state.play = not st.session_state.play

    if not st.session_state.play:
        insert_photos_buttons(l2, num_of_photos)
        str_frame = l3[1].text_input("Choose specific frame", str(st.session_state.count))
        if str_frame.isnumeric():  # Check if the data is numeric.
            st.session_state.count = int(str_frame)
            check_count_vaidation()
        st.session_state.count = l3[0].slider("Choose a frame", 0, num_of_photos - 1, st.session_state.count)
        str_steps = l2[3].text_input("steps", str(st.session_state.steps))
        if str_steps.isnumeric():
            st.session_state.steps = int(str_steps)
        if sort_key != st.session_state.prev_sort_key:
            st.session_state.organized_data = get_organized_data(num_of_photos, sort_key)
        show_analysis(is_cm_available, l)
        show_charts(st.session_state.organized_data, sort_key)
        l2[1].write("Frame ID is: " + str(st.session_state.frames_dict[st.session_state.count]))

        l[0].image(st.session_state.analyzeViewer.get_drawn_image(st.session_state.frames_dict[st.session_state.count]),
                   # Put the image on the screen.
                   width=700)
        st.session_state.prev_sort_key = sort_key
    else:
        st.session_state.play = not st.session_state.play
        x = l[0].empty()
        y = l2[5].empty()
        z = l2[1].empty()
        while not y.button("Stop", key=st.session_state.key_counter):
            x.image(
                st.session_state.analyzeViewer.get_drawn_image(st.session_state.frames_dict[st.session_state.count]),
                width=700)
            z.write("Frame ID is: " + str(st.session_state.frames_dict[st.session_state.count]))
            time.sleep(DELAY_BETWEEN_PICS)
            st.session_state.count += 1
            st.session_state.key_counter += 1
            check_count_vaidation()


def insert_photos_buttons(buttons, num_of_photos) -> None:
    if buttons[0].button("Back"):
        st.session_state.count -= st.session_state.steps
    if buttons[2].button("Forward"):
        st.session_state.count += st.session_state.steps
    check_count_vaidation()


def check_count_vaidation():
    if st.session_state.count < 0:
        st.session_state.count = st.session_state.num_of_photos - 1
    elif st.session_state.count > st.session_state.num_of_photos - 1:
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


def separate_graph_list(data, key):
    ret = []
    for d in data:
        tmp = []
        if "False Detection Rate Vs GT" in key:
            tmp.append(d.false_detection_rate_vs_gt)
        if "False Detection Rate Vs Pred" in key:
            tmp.append(d.false_detection_rate_vs_pred)
        if "FDR" in key:
            tmp.append(d.fdr)
        if "Miss Detection Rate" in key:
            tmp.append(d.mdr)
        if "Precision" in key:
            tmp.append(d.precision)
        if "Recall" in key:
            tmp.append(d.recall)

        ret.append(tmp)
    return ret


def sort_images(data, num_of_photos, key):
    if isinstance(key, list):
        func = ret_id
    elif key == "Recall":
        func = ret_recall
    elif key == "Precision":
        func = ret_pre
    elif key == "Miss Detection Rate":
        func = ret_mdr
    elif key == "False Detection Rate Vs GT":
        func = ret_fdrvg
    elif key == "False Detection Rate Vs Pred":
        func = ret_fdrvp
    else:
        func = ret_fdr
    sorted_data = sorted(data, key=func, reverse=st.session_state.reverse)
    for x in range(num_of_photos):
        st.session_state.frames_dict[x] = sorted_data[x].id
    return sorted_data


def ret_id(tmp): return tmp.id


def ret_recall(tmp): return tmp.recall, tmp.id


def ret_pre(tmp): return tmp.precision, tmp.id


def ret_fdr(tmp): return tmp.fdr, tmp.id


def ret_mdr(tmp): return tmp.mdr, tmp.id


def ret_fdrvg(tmp): return tmp.false_detection_rate_vs_gt, tmp.id


def ret_fdrvp(tmp): return tmp.false_detection_rate_vs_pred, tmp.id


def show_charts(data, key):
    if len(key) > 0:
        if isinstance(key, list):
            st.line_chart(pandas.DataFrame(data, columns=sorted(key)), height=400)
        else:
            st.line_chart(pandas.DataFrame(data), height=400)


def get_organized_data(num_of_photos, sort_key):
    graph_data = st.session_state.analyzeViewer.get_all_frames(num_of_photos)
    graph_data = sort_images(graph_data, num_of_photos, sort_key)
    return separate_graph_list(graph_data, sort_key)
