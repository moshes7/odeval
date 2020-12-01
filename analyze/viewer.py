import param
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn

from analyze.analyzer import Analyzer
from analyze.confusion_matrix import ConfusionMatrix
from analyze.visualize import convert_pyplot_figure_to_ndarray

class AnalyzerViewer(param.Parameterized):

    frame_id = param.Selector(label='Frame id', precedence=0.1)

    # metrics_table = param.DataFrame()
    iou_th = param.Number(0.5, bounds=(0, 1), precedence=0.2)
    score_th = param.Number(0.3, bounds=(0, 1), precedence=0.2)
    bbox_match_method = param.Selector(objects=['iou', 'pred_bbox_center'], precedence=0.2)

    def __init__(self, analyzer, resize_factor=0.5):

        super(AnalyzerViewer, self).__init__()

        analyzer.evaluate_performance()

        self.analyzer = analyzer
        self.resize_factor = resize_factor

        self.frame_ids = list(analyzer.keys())
        self.frame_ids.sort()
        self.frame_id = self.frame_ids[0]
        self.set_frame_ids()

        self.metrics_tables = None
        self.metrics_tables_total = None

        # axes handles
        self.ax_cm = None
        self.ax_cm_total = None
        self.ax_image_with_boxes = None

        # total performance
        self.fig_cm_total = self.plot_total_confusion_matrix2()

        # set thresholds
        self.iou_th = self.analyzer.iou_th
        self.score_th = self.analyzer.score_th
        self.bbox_match_method = self.analyzer.bbox_match_method


    @param.depends('frame_id')
    def set_frame_ids(self):
        self.param.frame_id.objects = self.frame_ids
        self.param.frame_id.default = self.frame_ids[0]
        self.param.frame_id.check_on_set = True


    @param.depends('frame_id', 'score_th', watch=False)
    def plot_image_with_boxes(self):

        self.analyzer.score_th = self.score_th

        image_with_boxes = self.analyzer.visualize_example(key=self.frame_id,
                                                           show_predictions=True,
                                                           show_ground_truth=True,
                                                           class_names=self.analyzer.class_names,
                                                           bgr2rgb=True,
                                                           resize_factor=self.resize_factor,
                                                           filter_pred_by_score=True,
                                                           )

        # TODO: try to improve speed by updating figure content, instead of creating new figure in every function call
        # create holoviews figure
        fig = hv.RGB(image_with_boxes)
        fig_width = image_with_boxes.shape[1] #// 20
        fig_height = image_with_boxes.shape[0] #// 20
        fig.options(width=fig_width, height=fig_height)
        fig = pn.pane.HoloViews(fig, width=fig_width, height=fig_height, sizing_mode='scale_both')

        # create matplotlib figure
        # ax = self.ax_image_with_boxes
        # if ax is None:
        #     plt.ioff()
        #     fig, ax = plt.subplots()
        # ax.clear()
        # ax.imshow(image_with_boxes)
        # title_str = '{}'.format(self.frame_id)
        # plt.title(title_str)
        # fig = ax.figure
        # fig.tight_layout()
        # self.ax_image_with_boxes = ax

        return fig


    @param.depends('frame_id', 'iou_th', 'score_th', 'bbox_match_method', watch=False)
    def plot_confusion_matrix(self):

        prediction, ground_truth, _, _, cm = self.analyzer.get_item_unpacked(self.frame_id)

        self.analyzer.bbox_match_method = self.bbox_match_method
        self.analyzer.iou_th = self.iou_th
        self.analyzer.score_th = self.score_th

        cm = ConfusionMatrix(prediction,
                             ground_truth,
                             self.analyzer.class_names,
                             bbox_match_method=self.analyzer.bbox_match_method,
                             iou_th=self.analyzer.iou_th,
                             score_th=self.analyzer.score_th,
                             add_miss_detection_col=self.analyzer.add_miss_detection_col,
                             add_false_detection_row=self.analyzer.add_false_detection_row,
                             )

        if self.analyzer.bbox_match_method == 'iou':
            title_str = 'Match Pred Method = IOU\nScore_th = {} | IOU_th = {}'.format(self.analyzer.score_th, self.analyzer.iou_th)
        elif self.analyzer.bbox_match_method == 'pred_bbox_center':
            title_str = 'Match Pred Method = Centers\nScore_th = {}'.format(self.analyzer.score_th, self.analyzer.bbox_match_method)
        ax, fig = ConfusionMatrix.plot_confusion_matrix(cm,
                                                        display_labels=self.analyzer.class_names,
                                                        add_miss_detection_col=self.analyzer.add_miss_detection_col,
                                                        add_false_detection_row=self.analyzer.add_false_detection_row,
                                                        title_str=title_str,
                                                        ax=self.ax_cm,
                                                        display=False,
                                                        )
        self.ax_cm = ax

        # convert to holoviews image
        # img = convert_pyplot_figure_to_ndarray(fig, dpi=180)
        # fig = hv.RGB(img)
        # fig.options(width=img.shape[1], height=img.shape[0])
        # fig = pn.pane.HoloViews(fig, width=250, height=200, sizing_mode='scale_both')

        try:
            self.metrics_tables = cm.metrics_tables
        except:
            self.metrics_tables = ConfusionMatrix.summarize_performance_metrics(cm.metrics, cm.class_names)

        return fig

    # @param.depends('frame_id', watch=False)
    # def get_metrics_table(self, metrics_type='global'):
    #
    #     self.metrics_table = self.metrics_tables[metrics_type]
    #
    #     return self.metrics_table


    @param.depends('frame_id', 'iou_th', 'score_th', 'bbox_match_method', watch=False)
    def get_global_metrics_table(self, metrics_type='global'):

        metrics_table = self.metrics_tables[metrics_type]

        matrics_table_widget = pn.widgets.DataFrame(metrics_table,
                                                    name='Gloabl Performance Metrics',
                                                    width=300)  #min_width=200, width_max=350, width_policy='fit')

        return matrics_table_widget


    @param.depends('frame_id', 'iou_th', 'score_th', 'bbox_match_method', watch=False)
    def get_class_metrics_table(self, metrics_type='class'):

        metrics_table = self.metrics_tables[metrics_type]

        matrics_table_widget = pn.widgets.DataFrame(metrics_table,
                                                   name='Per Class Performance Metrics',
                                                   min_width=300, width_policy='fit')

        return matrics_table_widget


    # @param.depends('iou_th', 'score_th', 'bbox_match_method', watch=False)
    # def plot_total_confusion_matrix(self):
    #     if self.analyzer.bbox_match_method == 'iou':
    #         title_str = 'Match Pred Method = IOU\nScore_th = {} | IOU_th = {}'.format(self.analyzer.score_th, self.analyzer.iou_th)
    #     elif self.analyzer.bbox_match_method == 'pred_bbox_center':
    #         title_str = 'Match Pred Method = Centers\nScore_th = {}'.format(self.analyzer.score_th, self.analyzer.bbox_match_method)
    #
    #     ax, fig = ConfusionMatrix.plot_confusion_matrix(self.analyzer.cm,
    #                                                     display_labels=self.analyzer.class_names,
    #                                                     add_miss_detection_col=self.analyzer.add_miss_detection_col,
    #                                                     add_false_detection_row=self.analyzer.add_false_detection_row,
    #                                                     title_str=title_str,
    #                                                     display=False,
    #                                                     )
    #
    #     return fig


    @param.depends('iou_th', 'score_th', 'bbox_match_method', watch=False)
    def plot_total_confusion_matrix2(self):

        # set analyzer thresholds
        self.analyzer.bbox_match_method = self.bbox_match_method
        self.analyzer.iou_th = self.iou_th
        self.analyzer.score_th = self.score_th

        # recalculate total confusion matrix
        self.analyzer.evaluate_performance(generate_report=False)

        cm = self.analyzer.cm

        if self.analyzer.bbox_match_method == 'iou':
            title_str = 'Match Pred Method = IOU\nScore_th = {} | IOU_th = {}'.format(self.analyzer.score_th, self.analyzer.iou_th)
        elif self.analyzer.bbox_match_method == 'pred_bbox_center':
            title_str = 'Match Pred Method = Centers\nScore_th = {}'.format(self.analyzer.score_th, self.analyzer.bbox_match_method)
        ax, fig = ConfusionMatrix.plot_confusion_matrix(cm,
                                                        display_labels=self.analyzer.class_names,
                                                        add_miss_detection_col=self.analyzer.add_miss_detection_col,
                                                        add_false_detection_row=self.analyzer.add_false_detection_row,
                                                        title_str=title_str,
                                                        ax=self.ax_cm_total,
                                                        display=False,
                                                        )
        self.ax_cm_total = ax

        try:
            self.metrics_tables_total = cm.metrics_tables
        except:
            self.metrics_tables_total = ConfusionMatrix.summarize_performance_metrics(self.analyzer.metrics, self.analyzer.class_names)

        return fig


    @param.depends('iou_th', 'score_th', 'bbox_match_method', watch=False)
    def get_global_metrics_table_total(self, metrics_type='global'):

        metrics_table = self.metrics_tables_total[metrics_type]

        matrics_table_widget = pn.widgets.DataFrame(metrics_table,
                                                    name='Gloabl Performance Metrics',
                                                    width=300)  #min_width=200, width_max=350, width_policy='fit')

        return matrics_table_widget


    @param.depends('iou_th', 'score_th', 'bbox_match_method', watch=False)
    def get_class_metrics_table_total(self, metrics_type='class'):

        metrics_table = self.metrics_tables_total[metrics_type]

        matrics_table_widget = pn.widgets.DataFrame(metrics_table,
                                                   name='Per Class Performance Metrics',
                                                   min_width=300, width_policy='fit')

        return matrics_table_widget


    def view(self, port=8081, show=True):
        # ---------------
        # define widgets
        # ---------------

        # define image with boxes
        # image_with_boxes_pn = pn.pane.Matplotlib(viewer.plot_image_with_boxes, dpi=288)

        tabs = pn.Tabs()

        # ------------------------------------
        # Total confusion matrix summary tab
        # ------------------------------------

        widgets1 = pn.Param(self.param, widgets={
            'bbox_match_method': {'type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
            'iou_th': {'type': pn.widgets.FloatSlider, 'step': 0.01},
            'score_th': {'type': pn.widgets.FloatSlider, 'step': 0.01},
        })

        # cm_total = pn.pane.Matplotlib(self.fig_cm_total, sizing_mode='scale_both', max_width=1000, dpi=300)

        # metrics_global_total = pn.widgets.DataFrame(self.analyzer.metrics_tables['global'],
        #                                             name='Gloabl Performance Metrics'
        #                                             , width=300)  #min_width=200, width_max=350, width_policy='fit')
        # metrics_class_total = pn.widgets.DataFrame(self.analyzer.metrics_tables['class'],
        #                                            name='Per Class Performance Metrics',
        #                                            min_width=300, width_policy='fit')
        # metrics_global_total = pn.widgets.DataFrame(self.get_global_metrics_table_total,
        #                                             name='Gloabl Performance Metrics'
        #                                             , width=300)  #min_width=200, width_max=350, width_policy='fit')
        # metrics_class_total = pn.widgets.DataFrame(self.get_class_metrics_table_total,
        #                                            name='Per Class Performance Metrics',
        #                                            min_width=300, width_policy='fit')

        c1 = pn.Column(#cm_total,
                       self.plot_total_confusion_matrix2,
                       pn.Spacer(height=5),
                       widgets1,
                       )
        # c2 = pn.Column(metrics_global_total, pn.Spacer(height=10), metrics_class_total)
        c2 = pn.Column(self.get_global_metrics_table_total,
                       pn.Spacer(height=10),
                       self.get_class_metrics_table_total)
        r1 = pn.Row(c1, pn.Spacer(width=10), c2)
        summary = r1

        tabs.extend([('Summary', summary)])

        # -------------------------------
        # Frames viewer tab
        # -------------------------------

        # define members widgets
        widgets = pn.Param(self.param, widgets={
            'frame_id': pn.widgets.DiscretePlayer,
            'bbox_match_method': {'type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
            'iou_th': {'type': pn.widgets.FloatSlider, 'step': 0.01},
            'score_th': {'type': pn.widgets.FloatSlider, 'step': 0.01},
        })

        c1 = pn.Column(self.plot_image_with_boxes, height_policy='fit', width_policy='max', max_width=1024)

        c2 = pn.Column(self.plot_confusion_matrix,
                       pn.Spacer(height=5),
                       self.param.frame_id,
                       pn.Spacer(height=5),
                       widgets,
                       )
        c3 = pn.Column(self.get_global_metrics_table,
                       pn.Spacer(height=10),
                       self.get_class_metrics_table
                       )

        r1 = pn.Row(c1, c2, pn.Spacer(width=50), c3)

        frames_viewer = r1

        tabs.extend([('Frames Viewer', frames_viewer)])

        tabs.servable()

        self.tabs = tabs

        # -----------
        # deploy app
        # -----------

        if show:  # if running outside of jupyter notebook
            # FIXME: should use better mechanism for selecting ports
            not_done = True
            while not_done:
                try:
                    self.tabs.show(port=port)
                except Exception as e:
                    if str(e) != '[Errno 98] Address already in use':
                        print(e)
                        not_done = False
                        raise
                    else:  # try next port
                        port += 1


if __name__ == '__main__':

    import os
    pn.extension()

    # load analyzer
    base_dir = os.path.dirname(__file__)
    relative_data_dir = './tests/data/ILSVRC2015_00078000'
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)

    # initialize viewer
    viewer = AnalyzerViewer(analyzer, resize_factor=0.5)

    # run viewer
    viewer.view(show=True)

    print('Done!')




