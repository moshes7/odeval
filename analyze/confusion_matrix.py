from __future__ import absolute_import, division, print_function, unicode_literals  # for python 2 compatibility

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import copy
import seaborn as sns

from analyze import visualize as vis
from analyze.visualize import convert_labels, rotate_xlabels
from analyze.bounding_box import Box

class ConfusionMatrix(object):

    """
    Class for computation, analysis and display of confusion matrix.
    Most of the core methods are implemented as static methods, so that they can be used stand alone.
    """

    def __init__(self,
                 prediction=None,
                 ground_truth=None,
                 class_names=None,
                 bbox_match_method='iou',
                 score_th=0.3,
                 iou_th=0.5,
                 add_miss_detection_col=True,
                 add_false_detection_row=True,
                 ):
        '''
        Initialize confusion matrix class.

        prediction : bounding_box.Box, optional
            Predictions Box object.
        ground_truth : bounding_box.Box, optional
            Ground truths Box object.
        class_names : list, optional
            List of class names. The order of names should correspond to class index at classifier output.
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
        add_miss_detection_col : bool, optional
            If True, another column will be added to confusion matrix, for specifying miss detections.
        add_false_detection_row : bool, optional
            If True, another row will be added to confusion matrix, for specifying false detections.
        '''

        self.prediction = copy.deepcopy(prediction)
        self.ground_truth = copy.deepcopy(ground_truth)
        self.class_names = class_names
        self.bbox_match_method = bbox_match_method
        self.score_th = score_th
        self.iou_th = iou_th
        self.add_miss_detection_col = add_miss_detection_col
        self.add_false_detection_row = add_false_detection_row
        self.cm = None
        self.metrics = None
        self.metrics_tables = None

        # verify that prediction and ground_truth use the same image shape
        if len(self.prediction) and (self.prediction.image_shape != self.ground_truth.image_shape):
            self.prediction.resize(self.ground_truth.image_shape)

        # calculate confusion matrix and performance metrics
        if (self.prediction is not None) and (self.ground_truth is not None) and (self.ground_truth is not None):
            self.simple_wrapper(self.prediction, self.ground_truth)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "bbox_match_method={}, ".format(self.bbox_match_method)
        s += "score_th={}, ".format(self.score_th)
        s += "iou_th={}, ".format(self.iou_th)
        s += "recall (macro)={}, ".format(self.metrics['global']['macro']['recall'] if self.metrics is not None else None)
        s += "precision (macro)={}".format(self.metrics['global']['macro']['precision'] if self.metrics is not None else None)
        s += ")"
        return s


    def simple_wrapper(self,
                       prediction,
                       ground_truth,
                       normalize=None,
                       display=False,
                       save_fig_name=None
                       ):

        '''
        Wrapper for simple confusion matrix calculation and analysis.

        prediction : bounding_box.Box, optional
            Predictions Box object.
        ground_truth : bounding_box.Box, optional
            Ground truths Box object.
        One of {None, 'gt', 'pred', 'all'}.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        display : bool, optional
            If True, confusion matrix will be displayed.
        save_fig_name : str, optional
            If not None, result image will be saved in this path.
        '''

        # calculate confusion matrix
        self.cm = ConfusionMatrix.calculate_confusion_matrix(prediction=copy.deepcopy(prediction),
                                                             ground_truth=copy.deepcopy(ground_truth),
                                                             class_names=self.class_names,
                                                             normalize=normalize,
                                                             bbox_match_method=self.bbox_match_method,
                                                             score_th=self.score_th,
                                                             iou_th=self.iou_th,
                                                             )

        # analyze
        self.metrics = ConfusionMatrix.analyze_performance(self.cm,
                                                           last_col_miss_detect=self.add_miss_detection_col,
                                                           last_row_false_detect=self.add_false_detection_row)

        self.metrics_tables = ConfusionMatrix.summarize_performance_metrics(self.metrics, self.class_names)

        # display
        if display or (save_fig_name is not None):
            if self.bbox_match_method == 'iou':
                title_str = 'Match Pred Method = IOU\nScore_th = {} | IOU_th = {}'.format(self.score_th, self.iou_th)
            elif self.bbox_match_method == 'pred_bbox_center':
                title_str = 'Match Pred Method = Centers\nScore_th = {}'.format(self.score_th, self.bbox_match_method)

            ConfusionMatrix.plot_confusion_matrix(self.cm,
                                                  display_labels=self.class_names,
                                                  add_miss_detection_col=self.add_miss_detection_col,
                                                  add_false_detection_row=self.add_false_detection_row,
                                                  title_str=title_str,
                                                  save_fig_name=save_fig_name,
                                                  display=display,
                                                  )



    @staticmethod
    def calculate_confusion_matrix(prediction,
                                   ground_truth,
                                   class_names,
                                   normalize=None,
                                   bbox_match_method='iou',
                                   score_th=0.3,
                                   iou_th=0.5,
                                   iou_criterion='all',
                                   score_criterion='all',
                                   add_miss_detection_column=True,
                                   add_false_detection_row=True,
                                   display=False,
                                   title_str='',
                                   ):
        '''
        Calculate confusion matrix, given ground truth and predictions bounding boxes.
        Add extra column for miss detections (ground truth box that was not predicted by any of the predictions),
        and extra row for false detection (prediction box that does not exist in the gt).

        prediction : bounding_box.Box
            Predictions Box object.
        ground_truth : bounding_box.Box
            Ground truths Box object.
        class_names : list, optional
            List of class names. The order of names should correspond to class index at classifier output.
        normalize : str, optional
            One of {'gt', 'pred', 'all'}
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.
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
        iou_criterion : str, optional
            TODO
        score_criterion : str, optional
            TODO
        add_miss_detection_column : bool, optional
            If True, another column will be added to confusion matrix, for specifying miss detections (i.e. ground truth
            boxes that do no correspond the any of the predictions).
        add_false_detection_row : bool, optional
            If True, another row will be added to confusion matrix, for specifying false predictions (i.e. prediction
            boxes that do no correspond the any of the ground truth).
        display : bool, optional
            If True, confusion matrix will be displayed.
        title_str : str, optional
            Title string.

        Returns
        -------
        cm : ndarray
            Confusion matrix.

        References:
            https://github.com/kaanakan/object_detection_confusion_matrix
            https://towardsdatascience.com/confusion-matrix-in-object-detection-with-tensorflow-b9640a927285
            skleran.metrics.confusion_matrix
        '''

        # TODO 1. add option of selecting prediction with ONLY best score and/or iou, or ALL predictions with scores and/or iou above th

        # ---------------------------
        # filter by score
        # ---------------------------
        if prediction.has_field('scores'):
            score_pred = prediction.get_field('scores')  # .tolist()

            # filter predictions by scores
            if score_criterion == 'all':  # consider all predictions with score higher than score_th
                # get indices of predictions with low scores
                indices_low_scores = np.where(score_pred < score_th)[0]
                indices_low_scores[::-1].sort()  # sort in descending order for proper deletion
                # delete unwanted predictions
                for ind in indices_low_scores:
                    del prediction[ind]

            elif score_criterion == 'best':  # consider only predictions with highest score
                # TODO: implement
                pass

        # ---------------------------
        # filter by iou
        # ---------------------------
        # initialize iou_indicator matrix
        iou_indicator = np.zeros((len(ground_truth), len(prediction)), dtype=np.int)  # shape of (num_gt, num_pred) ; 1 where iou exceeds threshold, 0 elsewhere
        if (len(ground_truth) > 0) and (len(prediction) > 0):  # calculate iou only if both objects are not empty
            # calculate iou matrix
            if bbox_match_method == 'iou':
                iou = Box.boxes_iou(ground_truth, prediction)  # shape of (num_gt, num_pred)
            elif bbox_match_method == 'pred_bbox_center':
                iou = Box.is_centers_inside_bbox(ground_truth, prediction)

            # filter predictions by iou
            if (bbox_match_method == 'pred_bbox_center'):
                    indices_high_iou = np.where(iou > iou_th) # greater than - to discard iou_th = 0
            elif (iou_criterion == 'all'):  # consider all predictions with iou higher than iou_th
                indices_high_iou = np.where(iou >= iou_th)  # greater than or equal to - to include iou_th = 0
            elif iou_criterion == 'best':  # consider only predictions with highest iou
                indices_high_iou = np.where(iou == np.max(iou, axis=1))

            # get wanted iou as an indicator matrix
            iou_indicator[indices_high_iou] = 1

        # ---------------------------
        # calculate confusion matrix
        # ---------------------------
        # get labels
        if len(ground_truth) > 0:
            labels_gt = ground_truth.get_field("labels")
            labels_gt = np.array(convert_labels(labels_gt, class_names, format_out='indices'))  # convert labels to indices

        if len(prediction) > 0:
            labels_pred = prediction.get_field("labels")
            labels_pred = np.array(convert_labels(labels_pred, class_names, format_out='indices'))  # convert labels to indices

        # initialization
        num_classes = len(class_names)
        num_rows = num_columns = num_classes
        if add_miss_detection_column:
            num_columns += 1  # add fictive class for miss detections
        if add_false_detection_row:
            num_rows += 1  # add fictive class for false detections
        cm_shape = (num_rows, num_columns)
        cm = np.zeros(cm_shape, dtype=np.float32)

        # treat TP, FP and miss detections:
        # iterate over ground truth
        for m in range(len(ground_truth)):
            # get gt label
            label_gt = labels_gt[m]

            # get iou of predictions correspond to this gt
            iou_row = iou_indicator[m, :]

            if add_miss_detection_column and (iou_row.sum() == 0):  # miss detection
                cm[label_gt, num_classes] += 1

            else:  # TP and FP (miss classification)
                # get prediction labels
                ind_pred = np.where(iou_row)[0]
                label_pred = labels_pred[ind_pred]

                # update confusion matrix
                cm[label_gt, label_pred] += 1

        # treat false detections:
        # iterate over predictions
        for n in range(len(prediction)):
            # get pred label
            label_pred = labels_pred[n]

            # get iou of gt correspond to this prediction
            iou_col = iou_indicator[:, n]

            if iou_col.sum() == 0:  # false detection - does not correspond to any of gt
                cm[num_classes, label_pred] += 1

        # normalize
        if normalize is not None:
            cm = ConfusionMatrix.normalize_confusion_matrix(cm, norm_type=normalize)
            title_str = '{}\nNormalize = {}'.format(title_str, normalize.upper())

        # display
        if display:
            display_labels = class_names.copy()  # use class names for labels
            ConfusionMatrix.plot_confusion_matrix(cm,
                                                  display_labels,
                                                  add_miss_detection_col=add_miss_detection_column,
                                                  add_false_detection_row=add_false_detection_row,
                                                  title_str=title_str,
                                                  )

        return cm

    @staticmethod
    def normalize_confusion_matrix(cm, norm_type='gt'):
        """
        Normalize confusion matrix.

        Parameters
        ----------
        cm : ndarray
            Confusion matrix of shape (n_classes, n_classes).
        norm_type : str
            One of {None, 'gt', 'pred', 'all'}.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.

        Returns
        -------
        cm : ndarray
            Normalized confusion matrix.
        """

        with np.errstate(all='ignore'):
            if norm_type == 'gt':
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif norm_type == 'pred':
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif norm_type == 'all':
                cm = cm / cm.sum()
            cm = np.nan_to_num(cm)

        return cm

    @staticmethod
    def plot_confusion_matrix(cm,
                              display_labels=None,
                              cmap=plt.cm.Blues,  # 'viridis',
                              xticks_rotation=35,  # 'vertical',  # 'horizontal',
                              ax=None,
                              add_miss_detection_col=True,
                              add_false_detection_row=True,
                              decimal_digits=2,
                              title_str='Confusion Matrix',
                              save_fig_name=None,
                              display=True,
                              ):
        """
        Plot confusion matrix.

        Parameters
        ----------
        cm : ndarray or ConfusionMatrix
            Confusion matrix of shape (n_classes, n_classes).
        display_labels : ndarray, optional
            Display labels for plot.
            If None, display labels are set from 0 to `n_classes - 1`.
        cmap : str or matplotlib Colormap, optional
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, optional
            Rotation of xtick labels.
        ax : matplotlib axes
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        add_miss_detection_col : bool, optional
            If True, another column label will be added to confusion matrix, for specifying miss detections.
        add_false_detection_row : bool, optional
            If True, another row label will be added to confusion matrix, for specifying false detections.
        decimal_digits : int, optional
            Number of decimal digits to which metrics will be rounded.
        title_str : str, optional
            Plot title.
        save_fig_name : str, optional
            If not None, result image will be saved in this path.
        display : bool, optional
            If True, image will be displayed on screen. Otherwise, image will be displayed, but can be saved to file.

        Returns
        -------
        ax, fig : axes and figure handles.

        References
        ----------
            skleran.metrics.plot_confusion_matrix
        """

        if not display:
            plt.ioff()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # clear figure
        for ax in fig.axes:
            try:
                ax.images[-1].colorbar.remove()
            except:
                pass
            try:
                ax.collections[-1].colorbar.remove()
            except:
                pass
            ax.clear()

        # take first axes
        ax = fig.axes[-1]

        if isinstance(cm, ConfusionMatrix):
            cm = cm.cm

        # round values
        cm = np.around(cm, decimal_digits)

        # set labels
        n_classes_y, n_classes_x = cm.shape[0], cm.shape[1]

        if display_labels is not None:
            display_labels_y = display_labels.copy()
            display_labels_x = display_labels.copy()
            if add_miss_detection_col:  # add miss detection label
                display_labels_x.append('miss det.')
            if add_false_detection_row:
                display_labels_y.append('false det.')

        else:
            display_labels_x = list(np.arange(n_classes_x))
            display_labels_y = list(np.arange(n_classes_y))
            # replace labels of last row/column
            if add_miss_detection_col:  # add miss detection label
                display_labels_x[-1] = 'miss det.'
            if add_false_detection_row:
                display_labels_y[-1] = 'false det.'

        ax = sns.heatmap(cm,
                         annot=True,
                         fmt='g',
                         annot_kws={'fontsize': 10,
                                    'verticalalignment': 'center'},
                         cmap=cmap,  # 'viridis',
                         linewidths=1.,
                         # linecolor='c',
                         xticklabels=display_labels_x,
                         yticklabels=display_labels_y,
                         ax=ax,
                         )
        # add title
        ax.set_title(title_str, fontsize=14)

        ax.set_ylabel(ylabel="Ground Truth", fontdict={'size': 13})
        ax.set_xlabel(xlabel="Prediction", fontdict={'size': 13})

        # fix rotated ticks display
        rotate_xlabels(fig, ax, rotation=xticks_rotation, bottom=0.)
        fig.tight_layout()

        # save figure
        if save_fig_name is not None:
            fig.savefig(save_fig_name, bbox_inches='tight')

        return ax, fig



    @staticmethod
    def plot_confusion_matrix_plt(cm,
                                  display_labels=None,
                                  include_values=True,
                                  xticks_rotation=35,  #'vertical',  # 'horizontal',
                                  values_format=None,
                                  cmap=plt.cm.Blues,  # 'viridis',
                                  ax=None,
                                  add_miss_detection_col=True,
                                  add_false_detection_row=True,
                                  decimal_digits=2,
                                  title_str='Confusion Matrix',
                                  save_fig_name=None,
                                  display=True,
                                  ):
        """
        Plot confusion matrix.

        Parameters
        ----------
        cm : ndarray or ConfusionMatrix
            Confusion matrix of shape (n_classes, n_classes).
        display_labels : ndarray, optional
            Display labels for plot.
            If None, display labels are set from 0 to `n_classes - 1`.
        include_values : bool, optional
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, optional
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, optional
            Rotation of xtick labels.
        values_format : str, optional
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.
        ax : matplotlib axes
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        add_miss_detection_col : bool, optional
            If True, another column label will be added to confusion matrix, for specifying miss detections.
        add_false_detection_row : bool, optional
            If True, another row label will be added to confusion matrix, for specifying false detections.
        decimal_digits : int, optional
            Number of decimal digits to which metrics will be rounded.
        title_str : str, optional
            Plot title.
        save_fig_name : str, optional
            If not None, result image will be saved in this path.
        display : bool, optional
            If True, image will be displayed on screen. Otherwise, image will be displayed, but can be saved to file.

        Returns
        -------
        ax, fig : axes and figure handles.

        References
        ----------
            skleran.metrics.plot_confusion_matrix
        """

        if not display:
            plt.ioff()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # clear figure
        try:
            ax.images[-1].colorbar.remove()
        except:
            pass
        ax.clear()

        if isinstance(cm, ConfusionMatrix):
            cm = cm.cm

        n_classes_y, n_classes_x = cm.shape[0], cm.shape[1]
        im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

        # display values of confusion matrix on cells centers
        if include_values:
            text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes_y), range(n_classes_x)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(np.around(cm[i, j], decimal_digits), 'g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        # add color bar
        fig.colorbar(im_, ax=ax)

        # add title
        # plt.suptitle(title_str, fontsize=16)
        plt.title(title_str, fontsize=14)

        # set labels
        if display_labels is not None:
            display_labels_y = display_labels.copy()
            display_labels_x = display_labels.copy()
            if add_miss_detection_col: # add miss detection label
                display_labels_x.append('miss det.')
            if add_false_detection_row:
                display_labels_y.append('false det.')

        else:
            display_labels_x = list(np.arange(n_classes_x))
            display_labels_y = list(np.arange(n_classes_y))
            # replace labels of last row/column
            if add_miss_detection_col:  # add miss detection label
                display_labels_x[-1] = 'miss det.'
            if add_false_detection_row:
                display_labels_y[-1] = 'false det.'

        # display labels
        ax.set(xticks=np.arange(n_classes_x),
               yticks=np.arange(n_classes_y),
               xticklabels=display_labels_x,
               yticklabels=display_labels_y,
               )

        ax.set_ylabel(ylabel="Ground Truth", fontdict={'size': 16})
        ax.set_xlabel(xlabel="Prediction", fontdict={'size': 16})

        # fix rotated ticks display
        fig.autofmt_xdate(rotation=xticks_rotation, bottom=0)

        # adjust figure
        # vis.pyplot_maximize_figure()
        fig.tight_layout()

        # save figure
        if save_fig_name is not None:
            fig.savefig(save_fig_name, bbox_inches='tight')

        return ax, fig


    @staticmethod
    def analyze_performance(cm, last_col_miss_detect=True, last_row_false_detect=True):
        """
        Calculate confusion matrix performance metrics.

        Parameters
        ----------
        cm : ndarray
            Confusion matrix of shape (n_classes, n_classes).
        last_col_miss_detect : bool, optional
            If True, the confusion matrix last column are used to specify miss detections.
        last_row_false_detect : bool, optional
            If True, the confusion matrix last row are used to specify false detections.

        Returns
        -------
        metrics : dict
            Evaluation metrics.
        """

        metrics_per_class = ConfusionMatrix.calc_per_class_metrics(cm, last_col_miss_detect, last_row_false_detect)
        metrics_global = ConfusionMatrix.calc_global_metrics(metrics_per_class)

        metrics = {'class': metrics_per_class,
                   'global': metrics_global,
                   }

        return metrics

    @staticmethod
    def calc_per_class_metrics(cm, last_col_miss_detect=True, last_row_false_detect=True, decimal_digits=2):
        """
        Calculate per class performance metrics.

        Parameters
        ----------
        cm : ndarray
            Confusion matrix of shape (n_classes, n_classes).
        last_col_miss_detect : bool, optional
            If True, the confusion matrix last column are used to specify miss detections.
        last_row_false_detect : bool, optional
            If True, the confusion matrix last row are used to specify false detections.
        decimal_digits : int, optional
            Number of decimal digits to which metrics will be rounded.

        Returns
        -------
        metrics : dict
            Per class performance metrics.

        References
        ----------
            Medium article: How to Improve Object Detection Evaluation.
            towards data science article: Multi-class Classification: Extracting Performance Metrics From The Confusion Matrix.
        """

        cm = cm.copy()

        # calculate basic metrics
        N_P = cm.sum(axis=0)  # number of predictions per class
        N_GT = cm.sum(axis=1)  # number of ground truth per class
        TP = np.diag(cm)  # true positives per class
        FP = N_P - TP  # false positives per class
        FN = N_GT - TP  # false negatives per class

        # treat last row and column
        miss_detection = np.zeros_like(N_GT) # initialization
        if last_col_miss_detect:
            miss_detection = cm[:, -1]
            # discard values calculated using last column (prediction related values)
            N_P = N_P[:-1]
            TP = TP[:-1]
            FP = FP[:-1]
            if last_row_false_detect:
                miss_detection = miss_detection[:-1]

        false_detection = np.zeros_like(N_P) # initialization
        if last_row_false_detect:
            false_detection = cm[-1, :]
            # discard values calculated using last row (ground truth related values)
            N_GT = N_GT[:-1]
            FN = FN[:-1]
            if last_col_miss_detect:
                false_detection = false_detection[:-1]

        # calculate population metrics
        N_P_total = N_P.sum()  # total number of predictions
        N_GT_total = N_GT.sum()  # total number of ground truth
        support = N_GT  # number of samples in each class

        # calculate selected performance metrics
        with np.errstate(all='ignore'):
            recall = np.nan_to_num(TP / N_GT)  # TPR, sensitivity, PD
            recall[N_GT == 0] = 0.  # 1.  # set recall of classes without GT to 1 (since recall = 0/0 we can chose either 0 or 1).
            precision = np.nan_to_num(TP / N_P)
            fdr = np.nan_to_num(FP / N_P)  # false discovery rate: FP out of predicted positive (FAR). equal to (1 - precision)
            miss_detection_rate = np.nan_to_num(miss_detection / N_GT)  # percentage of gt objects that were not detectet (out of all gt)
            false_detection_rate_vs_gt = np.nan_to_num(false_detection / N_GT)  # percentage of false detection, out of all gt
            false_detection_rate_vs_pred = np.nan_to_num(false_detection / N_P)  # percentage of false detection, out of all predictions

        # save metrics in dict

        metrics = {}
        # selected metrics
        metrics['recall'] = recall
        metrics['precision'] = precision
        metrics['fdr'] = fdr
        metrics['miss_detection_rate'] = miss_detection_rate
        metrics['false_detection_rate_vs_gt'] = false_detection_rate_vs_gt
        metrics['false_detection_rate_vs_pred'] = false_detection_rate_vs_pred
        # basic metrics
        metrics['TP'] = TP
        metrics['FP'] = FP
        metrics['FN'] = FN
        metrics['miss_detection'] = miss_detection
        metrics['false_detection'] = false_detection
        # population numbers
        metrics['N_P'] = N_P
        metrics['N_GT'] = N_GT
        metrics['N_P_total'] = N_P_total
        metrics['N_GT_total'] = N_GT_total
        metrics['support'] = support

        # round metrics
        if decimal_digits is not None:
            metrics = {key: np.around(val, decimal_digits) for key, val in metrics.items()}

        return metrics

    @staticmethod
    def calc_global_metrics(metrics_per_class, decimal_digits=2):
        """
        Calculate global class performance metrics.

        There are several ways to extend binary to multi-class classification. Here we use macro averaging, in which
        the statistics are calculated for each class separately, and then averaged over all classes, giving equal weight
        to each class.
        In addition, we calculate micro averaging, in which basic metrics (such as TP, FP, FN, etc.) are extracted and
        aggregated, then global performance metrics (such as precision, recall, etc.) are re-calculated using the
        aggregated values.

        Parameters
        ----------
        metrics_per_class : dict
            Per class performance metrics.
        decimal_digits : int, optional
            Number of decimal digits to which metrics will be rounded.

        Returns
        -------
        metrics_global : dict
            Global performance metrics - macro and micro.

        References
        ----------
        scikit-learn.org/stable/modules/model_evaluatoin.html#classification-metrics - Section '3.3.2.1 From binary to multiclass and multilabel'.
        StackExchange: What is the formula to calculate the precision, recall, f-measure with macro, micro, none for multilabel classification in sklearn metrics?
        """

        metrics_global = {}

        # macro averaging - consider only classes with elements
        N_P_nonzero_classes = np.count_nonzero(metrics_per_class['N_P'])  # number of classes with non zero predicted positive elements
        N_GT = metrics_per_class['N_GT']  # number of classes with ground truth positive elements
        N_GT_nonzero_classes = np.count_nonzero(N_GT)  # number of classes with non zero ground truth positive elements
        ind_N_GT_nonzero = np.nonzero(N_GT)
        metrics_global['macro'] = {}
        with np.errstate(all='ignore'):
            metrics_global['macro']['recall'] = np.nan_to_num(metrics_per_class['recall'][ind_N_GT_nonzero].sum() / N_GT_nonzero_classes) if N_GT_nonzero_classes > 0 else 1.
            metrics_global['macro']['precision'] = np.nan_to_num(metrics_per_class['precision'].sum() / N_P_nonzero_classes)  # FIXME: should we divide by N_P_nonzero_classes or N_GT_nonzero_classes ?
            metrics_global['macro']['fdr'] = np.nan_to_num(metrics_per_class['fdr'].sum() / N_P_nonzero_classes)  # FIXME: should we divide by N_P_nonzero_classes or N_GT_nonzero_classes ?
            metrics_global['macro']['miss_detection_rate'] = np.nan_to_num(metrics_per_class['miss_detection_rate'].sum() / N_GT_nonzero_classes)
            metrics_global['macro']['false_detection_rate_vs_gt'] = np.nan_to_num(metrics_per_class['false_detection_rate_vs_gt'].sum() / N_GT_nonzero_classes)
            metrics_global['macro']['false_detection_rate_vs_pred'] = np.nan_to_num(metrics_per_class['false_detection_rate_vs_pred'].sum() / N_P_nonzero_classes)  # FIXME: should we divide by N_P_nonzero_classes or N_GT_nonzero_classes ?
            metrics_global['macro']['N_GT_nonzero_classes'] = np.nan_to_num(N_GT_nonzero_classes)  # FIXME: should we divide by N_P_nonzero_classes or N_GT_nonzero_classes ?
            metrics_global['macro']['N_P_nonzero_classes'] = np.nan_to_num(N_P_nonzero_classes)  # FIXME: should we divide by N_P_nonzero_classes or N_GT_nonzero_classes ?

        # micro averaging
        TP = metrics_per_class['TP'].sum()
        FP = metrics_per_class['FP'].sum()
        FN = metrics_per_class['FN'].sum()
        miss_detection = metrics_per_class['miss_detection'].sum()
        false_detection = metrics_per_class['false_detection'].sum()
        N_P = metrics_per_class['N_P_total']
        N_GT = metrics_per_class['N_GT_total']

        with np.errstate(all='ignore'):
            recall = np.nan_to_num(TP / N_GT) if N_GT > 0 else 1.  # set recall of classes without GT to 1 (since recall = 0/0 we can chose either 0 or 1).
            precision = np.nan_to_num(TP / N_P)
            fdr = np.nan_to_num(FP / N_P)
            miss_detection_rate = np.nan_to_num(miss_detection / N_GT)
            false_detection_rate_vs_gt = np.nan_to_num(false_detection / N_GT)
            false_detection_rate_vs_pred = np.nan_to_num(false_detection / N_P)

        metrics_global['micro'] = {}
        metrics_global['micro']['TP'] = TP
        metrics_global['micro']['FP'] = FP
        metrics_global['micro']['FN'] = FN
        metrics_global['micro']['miss_detection'] = miss_detection
        metrics_global['micro']['false_detection'] = false_detection
        metrics_global['micro']['N_P'] = N_P
        metrics_global['micro']['N_GT'] = N_GT
        metrics_global['micro']['recall'] = recall
        metrics_global['micro']['precision'] = precision
        metrics_global['micro']['fdr'] = fdr
        metrics_global['micro']['miss_detection_rate'] = miss_detection_rate
        metrics_global['micro']['false_detection_rate_vs_gt'] = false_detection_rate_vs_gt
        metrics_global['micro']['false_detection_rate_vs_pred'] = false_detection_rate_vs_pred

        # round metrics
        if decimal_digits is not None:
            metrics_global['macro'] = {key: np.around(val, decimal_digits) for key, val in metrics_global['macro'].items()}
            metrics_global['micro'] = {key: np.around(val, decimal_digits) for key, val in metrics_global['micro'].items()}

        return metrics_global

    @staticmethod
    def summarize_performance_metrics(metrics, class_names=None):
        """
        Summarize performance metrics in tables.

        Parameters
        ----------
        metrics : dict
            Performance metrics.

        Returns
        -------
        performance_tables : dict
            Dictionary of pandas.DataFrame tables, which summarize performance.

        """

        # convert metrics to dataframes
        df_global = pd.DataFrame.from_dict(metrics['global'])
        df_class = pd.DataFrame.from_dict(metrics['class'])

        # --------------------
        # treat global metrics
        # --------------------
        # rearrange data frame for display
        # rename rows
        df_global.rename({'N_GT_nonzero_classes': '# classes with nonzero GT',
                          'N_P_nonzero_classes': '# classes with nonzero predictions',
                          'N_P': '# total predictions',
                          'N_GT': '# total GT',
                          'miss_detection_rate': 'miss detection rate',
                          'false_detection_rate_vs_gt': 'false detection rate (vs. GT)',
                          'false_detection_rate_vs_pred': 'false detection rate (vs. pred)',
                          },
                         inplace=True)
        # delete unwanted rows
        df_global.drop(index=['TP', 'FP', 'FN', 'miss_detection', 'false_detection'], inplace=True)

        # --------------------
        # treat class metrics
        # --------------------
        # add class names as first column
        if class_names is not None:
            df_class.insert(loc=0, column='class', value=class_names)

        # rearrange data frame for display
        # rename columns
        df_class.rename(columns={'N_P': '# predictions',
                                 'N_GT': '# GT'},
                        inplace=True)
        # delete unwanted columns
        df_class.drop(columns=['TP', 'FP', 'FN', 'miss_detection', 'false_detection',
                               'N_P_total', 'N_GT_total', 'support'], inplace=True)

        # pack tables in a dict
        performance_tables = {'global': df_global,
                              'class': df_class,
                              }

        return performance_tables