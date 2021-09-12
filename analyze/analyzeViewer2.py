import matplotlib.figure

from analyze.confusion_matrix import ConfusionMatrix
import streamlit as st
import holoviews as hv


class AnalyzeViewer2:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.ax_cm_total = None

    def get_confusion_matrix_for_frame(self, frame_id):
        prediction, ground_truth, _, _, cm = self.analyzer.get_item_unpacked(frame_id)

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

    def get_total_plot_cm(self):
        self.analyzer.evaluate_performance()
        ax, fig = ConfusionMatrix.plot_confusion_matrix(self.analyzer.cm,
                                                        display_labels=self.analyzer.class_names,
                                                        add_miss_detection_col=self.analyzer.add_miss_detection_col,
                                                        add_false_detection_row=self.analyzer.add_false_detection_row,
                                                        ax=self.ax_cm_total,
                                                        display=False,
                                                        )
        self.ax_cm_total = ax
        return fig

    def get_all_frames(self, size: int):
        data: list = []
        for i in range(size):
            tmp = self.get_confusion_matrix_for_frame(st.session_state.frames_dict[i])
            data.append([tmp.metrics["global"][st.session_state.micro_or_macro]["recall"], tmp.metrics["global"][st.session_state.micro_or_macro]["precision"], tmp.metrics["global"][st.session_state.micro_or_macro]["fdr"], st.session_state.frames_dict[i]])
        return data



