import os
import numpy as np

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


def basic_usage_example():

    """
    Simulate inference loop by using saved analyzer.
    """


    # load reference analyzer
    base_dir = os.path.dirname(__file__)
    relative_data_dir = '../tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer_ref = Analyzer.load(analyzer_file, load_images_from_dir=False)
    analyze_root_dir = os.path.abspath(os.path.join(base_dir, '..'))

    # create new analyzer
    output_dir = os.path.join(base_dir, 'output','simple_usage_example')
    analyzer = Analyzer(output_dir=output_dir,
                        output_video_name='video.avi',
                        class_names=CLASS_NAMES,
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
        prediction, ground_truth, image_path, image, _ = analyzer.unpack_item(item)
        image_path = os.path.join(analyze_root_dir, image_path)

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
                                                      class_names=CLASS_NAMES,
                                                      rgb2bgr=False,
                                                      display=False,
                                                      save_fig_path=image_out_path,
                                                      )[0]

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