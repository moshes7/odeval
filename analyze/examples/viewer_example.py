import os
from analyze.analyzer import Analyzer
from analyze.viewer import AnalyzerViewer
import analyze.viewer2
import streamlit as st


@st.cache(show_spinner=False)
def load_analzyer(path):
    base_dir = os.path.dirname(__file__)
    relative_data_dir = '../tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, path)
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)
    os.chdir('..')  # go one level up

    for i in range(len(analyzer.data)):
        analyzer.data[i][
            "image_path"] = "C:\\Users\\AMITNIM\\PycharmProjects\\odeval\\analyze\\tests\\data\\ILSVRC2015_00078000" \
                            "\\images\\" + os.path.basename( analyzer.data[i]["image_path"])
    return analyzer


if __name__ == '__main__':
    #analyzer = load_analzyer('analyzer.p')
    analyzer  = 0
    # viewer = AnalyzerViewer(analyzer, resize_factor=2.)
    # viewer.view()
    analyze.viewer2.run(analyzer)

    print('Done!')
