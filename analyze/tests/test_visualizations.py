import os
import cv2
import numpy as np
import pytest

from analyze.analyzer import Analyzer

CLASS_NAMES = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']

CLASS_NAME_2_IND = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
CLASS_IND_2_NAME = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))


def test_visualize_example():

    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyze_root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    os.chdir(analyze_root_dir)

    # load analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=True)

    # visualize example
    save_fig_path = os.path.join(base_dir, 'save_fig_example.png')
    frame_id = 40
    display = False  # True
    image = analyzer.visualize_example(key=frame_id, class_names=CLASS_NAMES, display=display, save_fig_path=save_fig_path)

    # compare to reference image
    image_ref_path = os.path.join(base_dir, 'data/visualizations/visualization_example.png')
    image_ref = cv2.imread(image_ref_path, cv2.IMREAD_UNCHANGED)

    is_close = np.isclose(image, image_ref, atol=2)  # allow up to 2 gray level difference

    is_all_close = np.all(is_close)

    assert is_all_close


if __name__ == '__main__':

    test_visualize_example()

    print('Done!')
