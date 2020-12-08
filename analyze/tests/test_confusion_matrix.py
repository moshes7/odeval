import os
import pytest
import numpy as np

from analyze.analyzer import Analyzer
from analyze.confusion_matrix import ConfusionMatrix
from analyze.bounding_box import Box


CLASS_NAMES = ['__background__', # always index 0
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


def test_confusion_matrix():

    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)

    # load analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    # initialize total confusion matrix
    num_classes = len(CLASS_NAMES)
    cm_total = np.zeros((num_classes + 1, num_classes + 1))

    # iterate over frames
    for frame_id, item in analyzer.items():

        # unpack data
        prediction, ground_truth, image_path, image = analyzer.unpack_item(item)[0:4]

        # calculate confusion matrix of current frame
        cm = ConfusionMatrix.calculate_confusion_matrix(prediction,
                                                        ground_truth,
                                                        CLASS_NAMES,
                                                        normalize=None,
                                                        score_th=0.3,
                                                        iou_th=0.5,
                                                        iou_criterion='all',
                                                        score_criterion='all',
                                                        display=False
                                                        )

        # analyze cm

        # add current cm to total cm
        cm_total += cm

    # analyze cm_total

    # normalize cm
    cm_total_norm = ConfusionMatrix.normalize_confusion_matrix(cm_total, norm_type='gt')
    # save_fig_name = os.path.join(data_dir, 'analysis/confusion_matrix.png')
    # ConfusionMatrix.plot_confusion_matrix(cm_total_norm, display_labels=CLASS_NAMES, save_fig_name=save_fig_name, display=False)  # for display

    cm_total_ref = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 159., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 37., 0., 0., 0., 0., 0., 0., 0., 30.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 11., 0., 11., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11., 0., 0., 0., 0., 0., 0., 0., 0.],
                             ])

    is_equal_mat = cm_total_ref == cm_total
    is_equal = cm_total == pytest.approx(cm_total_ref)

    assert is_equal


def test_calc_per_class_metrics():

    # define dummy confusion matrix
    cm = np.array([[10, 1 , 2 , 3],
                   [0 , 8 , 1 , 0],
                   [1 , 0 , 5 , 10],
                   [0 , 1 , 2 , 0]],
                  dtype=np.float32,
                  )

    # for debug - plot confusion matrix
    # ConfusionMatrix.plot_confusion_matrix(cm)

    # calculate basic metrics
    metrics = ConfusionMatrix.calc_per_class_metrics(cm)

    # calculate selected metrics by hand
    TP = np.array([10, 8, 5], dtype=np.float32)
    FP = np.array([1, 1+1, 2+1+2], dtype=np.float32)
    FN = np.array([1+2+3, 1, 1+10], dtype=np.float32)
    miss_detection = np.array([3, 0, 10], dtype=np.float32)
    false_detection = np.array([0, 1, 2], dtype=np.float32)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    recall[(TP + FN) == 0] = 0.
    fdr = 1 - precision  # FP / (TP + FP)
    miss_detection_rate = miss_detection / (TP + FN)
    false_detection_rate_vs_gt = false_detection / (TP + FN)
    false_detection_rate_vs_pred = false_detection / (TP + FP)

    # verify correctness
    assert TP == pytest.approx(metrics['TP'])
    assert FP == pytest.approx(metrics['FP'])
    assert FN == pytest.approx(metrics['FN'])
    assert miss_detection == pytest.approx(metrics['miss_detection'])
    assert false_detection == pytest.approx(metrics['false_detection'])
    assert precision == pytest.approx(metrics['precision'], abs=0.01)
    assert recall == pytest.approx(metrics['recall'], abs=0.01)
    assert fdr == pytest.approx(metrics['fdr'], abs=0.01)
    assert miss_detection_rate == pytest.approx(metrics['miss_detection_rate'], abs=0.01)
    assert false_detection_rate_vs_gt == pytest.approx(metrics['false_detection_rate_vs_gt'], abs=0.01)
    assert false_detection_rate_vs_pred == pytest.approx(metrics['false_detection_rate_vs_pred'], abs=0.01)


def test_calc_global_metrics():

    # define dummy confusion matrix
    cm = np.array([[10, 1 , 2 , 3],
                   [0 , 8 , 1 , 0],
                   [1 , 0 , 5 , 10],
                   [0 , 1 , 2 , 0]],
                  dtype=np.float32,
                  )

    # for debug - plot confusion matrix
    # ConfusionMatrix.plot_confusion_matrix(cm)

    # calculate metrics
    metrics_per_class = ConfusionMatrix.calc_per_class_metrics(cm)
    metrics_global = ConfusionMatrix.calc_global_metrics(metrics_per_class)

    # calculate several global metrics by hand

    # macro averaging
    precision = metrics_per_class['precision'].mean()
    recall = metrics_per_class['recall'].mean()
    miss_detection_rate = metrics_per_class['miss_detection_rate'].mean()
    false_detection_rate_vs_gt = metrics_per_class['false_detection_rate_vs_gt'].mean()

    assert precision == pytest.approx(metrics_global['macro']['precision'], abs=0.01)
    assert recall == pytest.approx(metrics_global['macro']['recall'], abs=0.01)
    assert miss_detection_rate == pytest.approx(metrics_global['macro']['miss_detection_rate'], abs=0.01)
    assert false_detection_rate_vs_gt == pytest.approx(metrics_global['macro']['false_detection_rate_vs_gt'], abs=0.01)

    # micro averaging
    TP = metrics_per_class['TP'].sum()
    FP = metrics_per_class['FP'].sum()
    FN = metrics_per_class['FN'].sum()
    N_GT = metrics_per_class['N_GT_total']
    N_P = metrics_per_class['N_P'].sum()

    recall = TP / (TP + FN)  # TP / N_GT
    precision = TP / (TP + FP)  # TP / N_P
    miss_detection_rate = metrics_per_class['miss_detection'].sum() / N_GT
    false_detection_rate_vs_pred = metrics_per_class['false_detection'].sum() / N_P

    assert precision == pytest.approx(metrics_global['micro']['precision'], abs=0.01)
    assert recall == pytest.approx(metrics_global['micro']['recall'], abs=0.01)
    assert miss_detection_rate == pytest.approx(metrics_global['micro']['miss_detection_rate'], abs=0.01)
    assert false_detection_rate_vs_pred == pytest.approx(metrics_global['micro']['false_detection_rate_vs_pred'], abs=0.01)


def test_empty_preds_gts():

    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)

    # load analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    # initialize total confusion matrix
    num_classes = len(CLASS_NAMES)
    cm_total = np.zeros((num_classes + 1, num_classes + 1))

    # iterate over frames
    counter = 0
    for frame_id, item in analyzer.items():

        # unpack data
        prediction, ground_truth, image_path, image = analyzer.unpack_item(item)[0:4]


        if counter == 1:  # delete predictions
            prediction = Box([], image_shape=(100, 200))
        elif counter == 2:  # delete groud truths
            ground_truth = Box([], image_shape=(100, 200))
        elif counter == 3:  # delete image shape
            try:
                ground_truth = Box([], image_shape=None)
            except ValueError as e:
                cond3 = e.__repr__() == "ValueError('image_shape must be tuple of length 2 (height, width) or 3 (height, width, channels), got None',)"

        # calculate confusion matrix of current frame
        cm = ConfusionMatrix.calculate_confusion_matrix(prediction,
                                                        ground_truth,
                                                        CLASS_NAMES,
                                                        normalize=None,
                                                        score_th=0.3,
                                                        iou_th=0.5,
                                                        iou_criterion='all',
                                                        score_criterion='all',
                                                        display=False
                                                        )

        if counter == 1:  # delete predictions
            cond1 = cm[:, -1].sum() == 3
        elif counter == 2:  # delete groud truths
            cond2 = cm[-1, :].sum() == 3

        # analyze cm

        # advance counter
        counter += 1

    # check conditions
    assert cond1
    assert cond2
    assert cond3


def test_initialize_with_simple_wrapper():

    base_dir = os.path.dirname(__file__)
    relative_data_dir = 'data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)

    # load analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    # initialize total confusion matrix
    num_classes = len(CLASS_NAMES)
    cm_total = np.zeros((num_classes + 1, num_classes + 1))

    # TODO: continue here, implement 2 options:
    # TODO: 1.calculate cm per image and aggregate total cm.
    # TODO: 2.calculate ground truths and predictions of all images, then calculate cm.

    # cm = ConfusionMatrix()


if __name__ == '__main__':

    test_confusion_matrix()
    test_calc_per_class_metrics()
    test_calc_global_metrics()
    test_empty_preds_gts()
    test_initialize_with_simple_wrapper()


    print('Done')