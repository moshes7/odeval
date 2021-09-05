import matplotlib.figure

from analyze.confusion_matrix import ConfusionMatrix
import streamlit as st
import holoviews as hv

class AnalyzeViewer2:
    def __init__(self, analyzer):
        self.frame_id = -1
        self.analyzer = analyzer

    @st.cache(show_spinner=False)
    def get_confusion_metrix_for_frame(self):
        prediction, ground_truth, _, _, cm = self.analyzer.get_item_unpacked(self.frame_id)

        return ConfusionMatrix(prediction,
                               ground_truth,
                               self.analyzer.class_names,
                               bbox_match_method=self.analyzer.bbox_match_method,
                               iou_th=self.analyzer.iou_th,
                               score_th=self.analyzer.score_th,
                               add_miss_detection_col=self.analyzer.add_miss_detection_col,
                               add_false_detection_row=self.analyzer.add_false_detection_row,
                               )

    def get_plot_cm(self, cm):
        return ConfusionMatrix.plot_confusion_matrix(cm,
                                                     display_labels=self.analyzer.class_names,
                                                     add_miss_detection_col=self.analyzer.add_miss_detection_col,
                                                     add_false_detection_row=self.analyzer.add_false_detection_row,
                                                     display=False,
                                                     )

    def get_drawn_image(self, frame_id):
        image_with_boxes = self.analyzer.visualize_example(key=frame_id,
                                                           show_predictions=True,
                                                           show_ground_truth=True,
                                                           class_names=self.analyzer.class_names,
                                                           bgr2rgb=True,
                                                           filter_pred_by_score=True,
                                                           )
        return image_with_boxes
