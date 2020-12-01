import os
import pytest
import numpy as np

from analyze.analyzer import Analyzer
from analyze.tests.utils_test import compare_instances


def test_analyzer_mutable_mapping_implementation():

    """
    Analyzer class inherits from collections.abc.MutableMapping, which demands implementation of
    several class methods such as __setitem__, __iter__, etc.
    Here we will check the correctness of these implementations.
    """

    # load analyzer
    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    analyzer2 = Analyzer()

    # check iteration
    for frame_id, item in analyzer.items():

        # check unpacking
        prediction, ground_truth, image_path, image = analyzer.unpack_item(item)[0:4]

        # check __setitem by assigning to second analyzer
        analyzer2[frame_id] = item


    # add attributes to analyzer2
    analyzer2.image_resize_factor = analyzer.image_resize_factor
    analyzer2.video_processor = analyzer.video_processor
    analyzer2.output_dir = analyzer.output_dir
    analyzer2.class_names = analyzer.class_names
    analyzer2.bbox_match_method = analyzer.bbox_match_method
    analyzer2.iou_th = analyzer.iou_th
    analyzer2.score_th = analyzer.score_th

    # compare 2 analyzers
    is_equal = compare_instances(analyzer, analyzer2)

    assert is_equal

    # check __delitem__
    keys = list(analyzer.keys())
    key_to_delete = keys[0]
    del analyzer[key_to_delete]


if __name__ == '__main__':

    test_analyzer_mutable_mapping_implementation()

    print('Done')