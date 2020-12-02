import pytest
import numpy as np
import matplotlib.pyplot as plt

from analyze.bounding_box import Box
import analyze.visualize as vis

def test_boxes_iou():

    # N = 3
    bbox1 = np.array([[0, 0, 9, 9],  # [left, top, right, bottom]
                      [0, 0, 4, 9],
                      [100, 100, 109, 109],
                      ])
    # M = 8
    bbox2 = np.array([[0, 5, 9, 9],
                      [0, 0, 19, 19],
                      [-5, -5, 4, 4],
                      [0, 0, 9, 9],
                      [5, 5, 9, 9],
                      [100, 100, 109, 109],
                      [95, 95, 104, 104],
                      [100, 100, 119, 119],
                     ])

    box1 = Box(bbox1, (1000, 1000))
    box2 = Box(bbox2, (1000, 1000))

    # calculate iou using my function
    iou = Box.boxes_iou(box1, box2)  # shape (N, M) = (3, 8)

    # ground truth - calculate by hand
    intersection_gt = np.array([[50, 100, 25, 100, 25, 0, 0, 0],
                                [25, 50, 25, 50, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 100, 25, 100],
                                ])

    area1 = np.array([100, 50, 100])
    area2 = np.array([50, 400, 100, 100, 25, 100, 100, 400])

    union_gt = area1[:, np.newaxis] + area2 - intersection_gt  # (3, 8)

    iou_gt = intersection_gt / union_gt

    iou_diff = iou_gt - iou

    is_equal_matrix = iou == iou_gt

    is_equal = iou == pytest.approx(iou_gt)

    assert is_equal


def test_is_points_inside_bounding_boxes():

    # shape (N, 4) = (3, 4)
    bbox1 = np.array([[100, 100, 200, 200],  # [left, top, right, bottom]
                      [500, 500, 700, 700],
                      [800, 900, 900, 999],
                      ])

    # shape (M, 4) = (6, 4)
    bbox2 = np.array([[30, 30, 90, 90],
                      [50, 50, 250, 250],
                      [300, 300, 400, 500],
                      [400, 400, 800, 800],
                      [490, 490, 600, 600],
                      [840, 840, 860, 860],
                     ])

    box1 = Box(bbox1, (1100, 1100))
    box2 = Box(bbox2, (1100, 1100))

    # for debug - display boxes
    display = False
    if display:
        class_names = ['1', '2', '3', '4', '5', '6']
        box1.add_field('labels', ['1', '2', '3'])
        box2.add_field('labels', ['1', '2', '3', '4', '5', '6'])
        image = 255 * np.ones((1100, 1100, 3), dtype=np.uint8) # gray image
        image, colors = vis.overlay_boxes(image, boxes=box1, class_names=class_names, color_factor=1, thickness=2)
        image = vis.overlay_scores_and_class_names(image, boxes=box1, class_names=class_names, colors=colors, text_size_factor=1.1, text_position='below')
        image, colors = vis.overlay_boxes(image, boxes=box2, class_names=class_names, color_factor=2, thickness=2)
        image = vis.overlay_scores_and_class_names(image, boxes=box2, class_names=class_names, colors=colors, text_size_factor=1.1)
        plt.figure()
        plt.imshow(image)  # convert to RGB and display
        centers = Box.bbox_centers(box2)
        for center in centers:
            plt.plot(center[0], center[1], '+', color='k')
        plt.show(block=False), plt.pause(1e-3)

    # calculate iou using my function
    is_center_inside = Box.is_centers_inside_bbox(bbox=box1, bbox2centers=box2)  # shape (N, M) = (3, 5)

    # ground truth - calculate by hand: shape (N, M) = (3, 6)
    is_center_inside_gt = np.array([[0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    ])

    is_equal_matrix = is_center_inside == is_center_inside_gt
    is_equal = is_center_inside == pytest.approx(is_center_inside_gt)
    assert is_equal


    # repeat the test for single dim box1 / box2
    bbox3 = bbox2[1, :]  # shape (1, ) => implicitly M = 1
    is_center_inside2 = Box.is_centers_inside_bbox(bbox=box1, bbox2centers=bbox3)  #
    is_center_inside_gt2 = np.array([[1],  # (N, M) = (3, 1)
                                     [0],
                                     [0],
                                     ])
    is_equal_matrix = is_center_inside2 == is_center_inside_gt2
    is_equal = is_center_inside == pytest.approx(is_center_inside_gt)
    assert is_equal

    bbox4 = bbox1[0, :]  # shape (1, ) => implicitly N = 1
    is_center_inside3 = Box.is_centers_inside_bbox(bbox=bbox4, bbox2centers=box2)
    is_center_inside_gt3 = np.array([[0, 1, 0, 0, 0, 0]])  # (N, M) = (1, 6)
    is_equal_matrix = is_center_inside3 == is_center_inside_gt3
    is_equal = is_center_inside == pytest.approx(is_center_inside_gt)
    assert is_equal


if __name__ == '__main__':

    test_boxes_iou()
    test_is_points_inside_bounding_boxes()

    print('Done')

