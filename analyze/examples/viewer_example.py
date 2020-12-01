import os

from analyze.analyzer import Analyzer
from analyze.viewer import AnalyzerViewer

if __name__ == '__main__':

    import os
    import panel as pn
    pn.extension()

    # load analyzer

    base_dir = os.path.dirname(__file__)
    relative_data_dir = '../tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)
    os.chdir('..')  # go one level up

    # initialize viewer
    viewer = AnalyzerViewer(analyzer, resize_factor=2.)

    # view analyzer
    viewer.view()

    print('Done!')
