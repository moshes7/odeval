from __future__ import absolute_import, division, print_function, unicode_literals  # for python 2 compatibility

from collections.abc import MutableMapping
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import dill
import json
import numpy as np
import copy
import pylatex
import time
import dataframe_image as dfi

from analyze import visualize as vis
from analyze.visualize import VideoProcessor
from analyze.utils import assign_dict_recursively
from analyze.confusion_matrix import ConfusionMatrix

class Analyzer(MutableMapping):
    """
    Class for saving and analyzing object detection results.
    This class inherits collections.abc.MutableMapping and behaves like dict.
    """

    def __init__(self,
                 image_resize_factor=None,
                 output_dir=None,
                 output_video_name=None,
                 class_names=None,
                 image_name_template='{:08d}.jpg',
                 # confusion matrix parameters
                 bbox_match_method='iou',
                 score_th=0.3,
                 iou_th=0.5,
                 add_miss_detection_col=True,
                 add_false_detection_row=True,
                 ):

        """
        Initialize Analyzer instance.
        By default, images directory is created, even if images will not be saved.

        The main data structure in Analyzer() is data. data is dict of dicts.
        The keys of the outer dict are data identifiers, such as frame numbers
        The inner dicts contains all the data needed for analyzing a single frame.
        Each of which contains the following keys:
            prediction : bounding_box.Box, optional
                Bounding box objeft.
            ground_truth : bounding_box.Box, optional
                Bounding box objeft.
            image_path : str, optional
                Path of image.
            image : ndarray, optional
                Image.

        Parameters
        ----------
        image_resize_factor : float, optional
            Resize factor to save image. Default is None (not used).
        output_dir : str, optional
            Output directory.
        output_video_name : str, optional
            If not None, contains path of output video.
            If None, video will not be saved.
        class_names : list, optional
            List of class names. The order of names should correspond to class index at classifier output.
        image_name_template : str, optional
            Template for image file names to be saved.
        bbox_match_method : str, optional
            Sets the method by which ground truth and prediction bounding boxes are matched.
            One of {'iou', 'pred_bbox_center'}:
                - 'iou': matching is done by IOU scores (and iou_th - see below).
                - 'pred_bbox_center': matching is done by checking if prediction bounding box centers is inside any of
                                      the ground truth bounding boxes.
        score_th : float, optional
            Score threshold, only predictions with scores higher than score_th will be considered.
        iou_th : float, optional
            IOU threshold, only predictions that overlap ground truth with iou higher than iou_th will be considered.
            Used only if bbox_match_method is 'iou'.

        Returns
        -------
        None
        """

        self.data = {}  # each entry should contain the keys [prediction, ground_truth, image_path, image]
        self.class_names = class_names

        self.image_resize_factor = image_resize_factor

        # confusion matrix parameters
        self.bbox_match_method = bbox_match_method
        self.score_th = score_th
        self.iou_th = iou_th
        self.add_miss_detection_col = add_miss_detection_col
        self.add_false_detection_row = add_false_detection_row

        # performance variables
        self.cm = None
        self.metrics = None
        self.metrics_tables = None

        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        # # TODO: currently not used
        # self.images_dir = os.path.join(output_dir, 'images')
        # os.makedirs(self.images_dir, exist_ok=True)
        # self.images_name_format = image_name_template
        # # TODO: end

        if output_video_name is not None:
            output_video_path = os.path.join(output_dir, output_video_name)
            self.video_processor = VideoProcessor(fps=12, output_video_path=output_video_path)
        else:
            self.video_processor = None

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def __setitem__(self, key, value):
        """
        Set analyzer key value pair.

        Parameters
        ----------
        key :
            Identifier of the data, such as frame number.
        value : dict
            Optionally contains the following keys:
                - prediction : bounding_box.Box
                - ground_truth : bounding_box.Box
                - image_path : str
                - image : ndarray

        Returns
        -------
        None
        """

        if key in self.keys():
            del self[key]

        # verify that value has all needed keys, assign None to keys that does not exist
        value_ref = {'prediction': None, 'ground_truth': None, 'image_path': None, 'image': None, 'cm': None}
        value = assign_dict_recursively(value_ref, value)  # add missing keys if necessary

        # resize image if necessary
        if value['image'] is not None:
            value['image'] = self.resize_image(value['image'])

        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_elements={}, ".format(len(self.data))
        s += "images_resize_factor={}".format(self.image_resize_factor)
        return s

    def update_analyzer(self,
                        key,
                        prediction=None,
                        ground_truth=None,
                        image_path=None,
                        image=None,
                        analyze_performance=False,
                        ):
        """
        Update Analyzer data.
        This method is equivalent to __setitem__ method, which enables to pass data as separate arguments instead
        of as a dictionary.

        Parameters
        ----------
        key :
            Data identifer.
        prediction : bounding_box.Box, optional
            Predictions Box object.
        ground_truth : bounding_box.Box, optional
            Ground truths Box object.
        image_path : str, optional
            Path of image.
        image : ndarray, optional
            Image.

        Returns
        -------
        None
        """

        # resize image
        if image is not None:
            image = self.resize_image(image)

        item_dict = {'prediction': prediction,
                     'ground_truth': ground_truth,
                     'image_path': image_path,
                     'image': image
                     }
        
        if analyze_performance:
            cm = ConfusionMatrix(prediction,
                                 ground_truth,
                                 self.class_names,
                                 bbox_match_method=self.bbox_match_method,
                                 iou_th=self.iou_th,
                                 score_th=self.score_th,
                                 add_miss_detection_col=self.add_miss_detection_col,
                                 add_false_detection_row=self.add_false_detection_row,
                                 )
            item_dict['cm'] = cm

        self.__setitem__(key, item_dict)

    def keys(self):
        return self.data.keys()

    def get_item_unpacked(self, key):

        item = self[key]

        prediction, ground_truth, image_path, image, cm = self.unpack_item(item)

        return prediction, ground_truth, image_path, image, cm

    def unpack_item(self, item):

        prediction = item['prediction']
        ground_truth = item['ground_truth']
        image_path = item['image_path']
        image = item['image']
        cm = item.get('cm', None)

        return prediction, ground_truth, image_path, image, cm


    def resize_image(self, image, images_resize_factor=None):
        """
        Update images.

        Parameters
        ----------
        image : ndarray
            Image.

        Returns
        -------
        None
        """

        if images_resize_factor is None:
            images_resize_factor = self.image_resize_factor

        if images_resize_factor is not None:
            image = cv2.resize(image, None, None, fx=self.image_resize_factor, fy=self.image_resize_factor)

        return image


    def export_analyzer_data(self, output_file, export_type, export_images=False, save_self=False):
        """
        Export analyzer data to file.

        Parameters
        ----------
        export_type : str
            Path of output file.
        export_type : str or list
            Export type, one of {'pickle', TODO: 'json', 'csv'}
            If a list is given, than all the export types will be saved
        export_images : bool, optional
            If True, images will be exported.

        Returns
        -------
        Box object.
        """

        # TODO: currently not used. should update code before usage.

        # cast export_type to list (to support multiple export types)
        if not isinstance(export_type, list):
            export_type = list(export_type)

        # verify that output directory exist, if not - create it
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # recursive

        # pack output data in a list
        data_out = [self.predictions, self.ground_truths, self.images_path]

        if export_images:  # may consume a lot of memory, avoid if not neccessary
            data_out.append(self.images)
            # TODO: maybe export images as png files

        if export_type == 'pickle':
            output_pickle = '{}.p'.format(output_file)  # add file extension
            with open(output_pickle, 'wb') as f:
                pickle.dump(data_out, f)

        if export_type == 'json':
            output_json = '{}.json'.format(output_file)  # add file extension
            with open(output_json, 'w') as f:
                json.dump(data_out, f, indent=4, sort_keys=True)  # sort keys for pretty print


    def save(self, output_file=None, save_thin_instance=True, save_images_in_dir=True, image_name_template='{:08d}.jpg'):
        """
        Save Analyzer instance.

        Parameters
        ----------
        output_file : str, optional
            Path for saving output file.
        save_thin_instance : bool, optional
            If True, instance will be saved without images, in order to save memory.
        save_images_in_dir : bool, optional
            If True, images will be saved in a sub-directory named 'images' in the same level of output_file.
            Images are saved using opencv, and color channels are assumed to be BGR.
        image_name_template : str, optional
           Template for images name to be saved.
           Default is frame number with leading zeros up to width 8, and jpg format

        Returns
        -------
        None
        """
        
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'analyzer.p')
        
        # verify that output directory exist, if not - create it
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # recursive

        # save images in dir
        if save_images_in_dir:
            images_dir = os.path.join(output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            for frame_id, item in self.items():
                try:
                    image_name = image_name_template.format(frame_id)  # use 8 leading zeros
                    cv2.imwrite(os.path.join(images_dir, image_name), item['image'])
                except:
                    pass  # no image saved in analyzer

        # make a copy of current analyzer instance
        analyzer_instance = dill.copy(self)

        # make thin instance - delete images from analyzer
        if save_thin_instance:
            for key in analyzer_instance.keys():
                analyzer_instance[key]['image'] = None

        # save object instance
        with open(output_file, 'wb') as f:
            dill.dump(analyzer_instance, f)


    @staticmethod
    def load(input_file, load_images_from_dir=False, sfx='jpg'):
        """
        Load Analyzer instance.

        Parameters
        ----------
        input_file : str
            Path of saved Analyzer file.
        load_images_from_dir : bool, optional
            If True, images will be loaded from sub directory named images.
        sfx : str, optional
           Suffix of images to be loaded, defines the image format. Default is 'jpg'.

        Returns
        -------
        None
        """

        # load saved instace
        with open(input_file, 'rb') as f:
            analyzer = dill.load(f)

        # update current instance by loaded instance
        # self.__dict__.updated(instance.__dict__)

        # load images from directory
        if load_images_from_dir:
            images_dir = os.path.join(os.path.dirname(input_file), 'images')
            image_list = os.listdir(images_dir)
            image_list = [image_name for image_name in image_list if image_name.endswith(sfx)]  # filter images using suffix
            for name in image_list:
                frame_id = int(name.split('.')[0])
                image_path = os.path.join(images_dir, name)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                analyzer[frame_id]['image'] = image

        return analyzer


    def visualize_example(self,
                          key,
                          image=None,
                          show_predictions=True,
                          show_ground_truth=True,
                          class_names=None,
                          rgb2bgr=False,
                          bgr2rgb=False,
                          display=False,
                          save_fig_path=None,
                          resize_factor=None,
                          filter_pred_by_score=False,
                          ):
        """
        Overlay predictions and ground truths on top of image.

        Parameters
        ----------
        key :
            Data identifer.
        image : ndarray, optional
            If not None, this image will be used.
            Otherwise, image will be read from current instance or from file.
        show_predictions : bool, optional
            If True, predicted boxes will be overlayed on top of image.
        show_ground_truth : bool, optional
            If True, ground truth boxes will be overlayed on top of image.
        class_names : list, optional
            List of class names. The order of names should correspond to class index at classifier output.
        rgb2bgr : bool, optional
            If True, image colors will be converted from RGB to BGR after overlaying boxes, before dispaly / save.
        display : bool, optional
            If True, image will be displayed.
        save_fig_path : str, optional
            If not None, image will be saved on this path.
        filter_pred_by_score : bool, optional
            If True, only predictions with score higher than self.score_th will be displayed.

        Returns
        -------
        image : ndarray
            Image with overlayed boxes.

        """

        # TODO: deal with scaled image by resize boxes as necessary

        prediction, ground_truth, image_path, image_from_memory, cm = self.get_item_unpacked(key)

        if image is None:
            image = image_from_memory

        # get image
        if image is None:  # if image is not None - use input image
            image_name = os.path.abspath(self[key]['image_path'])
            image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            image = vis.normalize_image_for_display(image, bgr2rgb=False)

        image_size_factor = (image.shape[0] + image.shape[1]) / 1500
        image_size_factor = np.clip(image_size_factor, 0.5, 1.5)

        # overlay ground truths
        if show_ground_truth and (ground_truth is not None) and (len(ground_truth) > 0):
            if ground_truth.image_shape != image.shape:
                ground_truth.resize(image.shape)
            image, colors = vis.overlay_boxes(image, boxes=ground_truth, class_names=class_names, color_factor=0, image_size_factor=image_size_factor)
            image = vis.overlay_scores_and_class_names(image, boxes=ground_truth, class_names=class_names, colors=colors, text_size_factor=0.7, text_position='below', image_size_factor=image_size_factor)

        # overlay predictions
        if show_predictions:
            # delete predictions with low scores
            if filter_pred_by_score:
                prediction = copy.deepcopy(prediction)  # make a copy before deletion
                score_pred = prediction.get_field('scores')  # .tolist()
                indices_low_scores = np.where(score_pred < self.score_th)[0]
                indices_low_scores[::-1].sort()  # sort in descending order for proper deletion
                # delete unwanted predictions
                for ind in indices_low_scores:
                    del prediction[ind]

            if len(prediction) > 0:
                if prediction.image_shape != image.shape:
                    prediction.resize(image.shape)
                image, colors = vis.overlay_boxes(image, boxes=prediction, class_names=class_names, color_factor=1, thickness=3, image_size_factor=image_size_factor)
                image = vis.overlay_scores_and_class_names(image, boxes=prediction, class_names=class_names, colors=colors, text_size_factor=0.8, image_size_factor=image_size_factor)

        # convert colors
        if rgb2bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if resize_factor is not None:
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)

        # display
        if display:
            plt.figure()
            if rgb2bgr:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert to RGB and display
            else:
                plt.imshow(image)
            title_str = '{}'.format(key)
            plt.title(title_str)
            # vis.pyplot_maximize_figure()
            plt.show(block=False), plt.pause(1e-3)

        # save figure
        if save_fig_path is not None:
            cv2.imwrite(save_fig_path, image)

        return image

    def update_video(self, images):
        """
        Add image to video.

        Parameters
        ----------
        images : nadarry
            Image to save.

        Returns
        -------
        None.
        """

        if not isinstance(images, list):
            images = list(images)

        # update video
        self.video_processor.write_frames(images)


    def evaluate_performance(self, generate_report=False, reevalute=True):
        """
        Evaluate analyzer performance by calculating total confusion matrix and selected performance metrics.

        Parameters
        ----------
        generate_report : bool, optional
            If True, performance pdf report will be generated.
        reevalute : bool, optional
            If True, confusion matrix will be calculated again with current analyzer parameters, regardless with data
            loaded from saved analyzer.


        Returns
        -------
        None.
        """
        # calculate total confusion matrix

        num_classes = len(self.class_names)
        num_rows = num_columns = num_classes
        if self.add_miss_detection_col:
            num_columns += 1  # add fictive class for miss detections
        if self.add_false_detection_row:
            num_rows += 1  # add fictive class for false detections
        cm_shape = (num_rows, num_columns)
        cm_total = np.zeros(cm_shape, dtype=np.float32)

        # iterate over frames
        for frame_id, item in self.items():
            # unpack data
            prediction, ground_truth, image_path, image, cm = self.unpack_item(item)

            if reevalute:
                cm = None
            else:
                cm = cm.cm  # get cm array from cm object

            # calculate confusion matrix of current frame
            if cm is None:
                prediction = copy.deepcopy(prediction)  # make a copy before deletion
                cm = ConfusionMatrix.calculate_confusion_matrix(prediction,
                                                                ground_truth,
                                                                class_names=self.class_names,
                                                                normalize=None,
                                                                bbox_match_method=self.bbox_match_method,
                                                                score_th=self.score_th,
                                                                iou_th=self.iou_th,
                                                                add_miss_detection_column=self.add_miss_detection_col,
                                                                add_false_detection_row=self.add_false_detection_row,
                                                                display=False
                                                                )

            # add current cm to total cm
            cm_total += cm

        # calculate performance metrics
        self.cm = cm_total
        self.metrics = ConfusionMatrix.analyze_performance(self.cm,
                                                           last_col_miss_detect=self.add_miss_detection_col,
                                                           last_row_false_detect=self.add_false_detection_row)
        self.metrics_tables = ConfusionMatrix.summarize_performance_metrics(self.metrics, class_names=self.class_names)

        # generate pdf report
        if generate_report:
            self.perfromance_report()


    def perfromance_report(self, save_pdf=False, pdf_name='analyzer_report'):
        """
        Generate performance report. By default, images of confusion matrix and performance metrics will be saved.
        If save_pdf is True, a pdf will also be saved.
        Output will be saved in self.output_dir

        Parameters
        ----------
        save_pdf : bool, optional
            If True, pdf report will be saved.
        pdf_name : str, optional
            PDF report file name, used only if save_pdf is True.

        Returns
        -------
        None.
        """

        # save confusion matrix
        if self.bbox_match_method == 'iou':
            title_str = 'Match Pred Method = IOU\nScore_th = {} | IOU_th = {}'.format(self.score_th, self.iou_th)
        elif self.bbox_match_method == 'pred_bbox_center':
            title_str = 'Match Pred Method = Centers\nScore_th = {}'.format(self.score_th, self.bbox_match_method)

        image_path_cm = os.path.join(self.output_dir, 'total confusion matrix.png')
        h_ax, h_fig = ConfusionMatrix.plot_confusion_matrix(self.cm,
                                                            display_labels=self.class_names,
                                                            add_miss_detection_col=self.add_miss_detection_col,
                                                            add_false_detection_row=self.add_false_detection_row,
                                                            title_str=title_str,
                                                            display=False,
                                                            save_fig_name=image_path_cm,
                                                            )

        # save performance metrics

        # text files
        self.metrics_tables['global'].to_csv(os.path.join(self.output_dir, 'global metrics.csv'))
        self.metrics_tables['class'].to_csv(os.path.join(self.output_dir, 'class metrics.csv'))

        # images
        image_path_class_metrics = os.path.join(self.output_dir, 'class metrics.png')
        image_path_global_metrics = os.path.join(self.output_dir, 'global metrics.png')
        dfi.export(self.metrics_tables['class'], image_path_class_metrics)
        dfi.export(self.metrics_tables['global'], image_path_global_metrics)

        # save pdf report
        if save_pdf:

            doc = pylatex.Document(geometry_options={'margin': '0.2in'})

            doc.preamble.append(pylatex.Command('title', 'Analyzer Report'))
            doc.preamble.append(pylatex.Command('date', pylatex.NoEscape(r'\today')))
            doc.append(pylatex.NoEscape(r'\maketitle'))

            # plot confusion matrix
            with doc.create(pylatex.Section('Confusion Matrix')):
                with doc.create(pylatex.Figure(position='h!')) as fig:

                    fig.add_image(image_path_cm, placement='center')

            # global metrics
            with doc.create(pylatex.Section('Performance Metrics')):
            # with doc.create(pylatex.Section('Global Metrics')):
                with doc.create(pylatex.Figure(position='h!')) as fig:
                    fig.add_image(image_path_global_metrics, width=200, placement='center')

            # class metrics
            # with doc.create(pylatex.Section('Class Metrics')):
                with doc.create(pylatex.Figure(position='h!')) as fig:
                    fig.add_image(image_path_class_metrics, placement='center')

            # save report
            output_full_name = '{}'.format(pdf_name)
            report_file = os.path.abspath(os.path.join(self.output_dir, output_full_name))
            doc.generate_pdf(report_file, clean_tex=True, compiler='pdflatex', silent=True)

        pass
