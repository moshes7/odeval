import os
import numpy as np

from analyze.analyzer import Analyzer
from analyze.tests.utils_test import load_test_analyzer


def basic_usage_example():

    """
    Simulate inference loop by using saved analyzer.
    """

    # load reference analyzer
    analyzer_ref, analyze_root_dir = load_test_analyzer()
    base_dir = os.path.dirname(__file__)
    images_path = os.path.abspath(os.path.join(base_dir, '..', 'tests/data/ILSVRC2015_00078000/images'))

    # create new analyzer
    output_dir = os.path.join(base_dir, 'output','simple_usage_example')
    analyzer = Analyzer(output_dir=output_dir,
                        output_video_name='video.avi',
                        class_names=analyzer_ref.class_names,
                        bbox_match_method='pred_bbox_center',
                        score_th=0.25,
                        iou_th=0.4,
                        )

    # set output images format and directory
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    output_image_format = 'jpg'  # png
    pattern_length = 8
    output_image_pattern = '%0{}d.{}'.format(pattern_length, output_image_format)
    images_with_boxes = []

    # simulate inference loop by iterating over data saved in analyzer_ref
    counter = 0
    for frame_id, item in analyzer_ref.items():

        # inference simulation
        prediction, ground_truth, image_path, image, _ = analyzer_ref.unpack_item(item)
        image_path = os.path.join(images_path, os.path.basename(image_path))

        # log results in analyzer
        # IMPORTANT:
        # in real usage of analyzer you will probably need to to some pre-processing to convert inference output to the
        # form that analyzer expects. mainly - convert predictions and ground_truth to bounding_box.Box().
        analyzer.update_analyzer(key=frame_id,
                                 prediction=prediction,
                                 ground_truth=ground_truth,
                                 image_path=image_path,
                                 analyze_performance=True)

        # optional: save visualizations
        image_out_path = os.path.join(output_images_dir, output_image_pattern % frame_id)
        image_with_boxes = analyzer.visualize_example(key=frame_id,
                                                      image=image,
                                                      show_predictions=True,
                                                      show_ground_truth=True,
                                                      class_names=analyzer_ref.class_names,
                                                      rgb2bgr=False,
                                                      display=False,
                                                      save_fig_path=image_out_path,
                                                      )

        # optional: save images with visualizations in a list - later will be used for video creation
        images_with_boxes.append(image_with_boxes)

        # optional: periodically save analyzer and video, can omit and save only at inference loop end
        if np.mod(counter, 20) == 0:
            analyzer.save(save_thin_instance=False, save_images_in_dir=True)
            analyzer.update_video(images_with_boxes)

        counter += 1

    # save final analyzer and video
    analyzer.save(save_thin_instance=True, save_images_in_dir=False)
    analyzer.update_video(images_with_boxes)

    # analyze full run - save performance report in output folder
    analyzer.evaluate_performance(generate_report=True)

    print('basic_usage_example - done!')



if __name__ == '__main__':

    basic_usage_example()

    print('Done!')