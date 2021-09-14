import os
from analyze.analyzer import Analyzer
import analyze.viewer2
import streamlit as st

if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.micro_or_macro = "macro"


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_analzyer(path):
    base_dir = os.path.dirname(__file__)
    relative_data_dir = '../tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, path)
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)
    for i in range(len(analyzer.data)):
        analyzer.data[i]["image_path"] = os.getcwd() + "/../tests/data/ILSVRC2015_00078000" \
                                                       "/images/" + os.path.basename(analyzer.data[i]["image_path"])
    return analyzer


if __name__ == '__main__':
    analyzer = load_analzyer('analyzer.p')
    analyze.viewer2.run(analyzer)
