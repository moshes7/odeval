import numpy as np
import cv2
import io
import copy
import matplotlib
import matplotlib.pyplot as plt


def normalize_image_uint8(img):
    """
    Normalize dynamic range to [0, 255] and cast type to np.uint8

        Parameters
        ----------
        img : ndarray
            Input image.

        Returns
        -------
        img : ndarray
            Output image
    """

    # cast image type to float
    img = np.array(img, dtype=np.float32)

    # normalize dynamic range to [0, 255]
    img -= img.min()
    img = img * 255. / img.max()

    # cast type
    img = img.astype(np.uint8)

    return img


def normalize_image_for_display(img, uint8=True, bgr2rgb=True, display=False, title=''):
    """
    Normalize image for display.

    Parameters
    ----------
    img : ndarray
        Input image.
    uint8 : bool, optional
        If True, image type will be casted to uint8 with dynamic range of [0, 255].
    bgr2rgb : bool, optional
        If True, color channels order will be reversed.
    display: bool, optional
        If True, image will be displayed on screen.
    title : str, optional
        Image title, relevant only for displayed images.


    Returns
    -------
    img_out : ndarray
        Output image
    """

    # make a copy
    img_out = copy.copy(np.array(img))

    # if channels are transposed as in torch tensor CHW - rearrange them to HWC
    if img.shape[0] == 3:  # CHW
        img_out = np.transpose(img_out, [1, 2, 0])  # HWC

    # convert BGR to RGB
    if bgr2rgb:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    if uint8:
        img_out = normalize_image_uint8(img_out)

    # display
    if display:
        plt.figure()
        plt.imshow(img_out)
        plt.title(title)
        # pyplot_maximize_figure()
        plt.show(block=False), plt.pause(1e-3)

    return img_out


def pyplot_maximize_figure():
    """
    Maximize matplotlib figure.
    """
    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'QtAgg':
        mng.window.showMaximized()


def class_names_indices_mapping(class_names):
    """
    Given list of class names, generate mapping from class names to class indices and vice versa.

    Parameters
    ----------
    class_names : list, optional
        List of class names. The order of names should correspond to class index at classifier output.

    Returns
    -------
    name_to_ind : dict
        Dictionary which maps from name to index.
    ind_to_name : dict
        Dictionary which maps from index to name.
    """

    name_to_ind = dict(zip(class_names, range(len(class_names))))
    ind_to_name = dict(zip(range(len(class_names)), class_names))

    return name_to_ind, ind_to_name


def convert_labels(labels, class_names, format_out):
    """
    Given list of labels as class names (str) or class indices (int), convert them to wanted output type.
    If labels is already in wanted format, it will not be changed

    Parameters
    ----------
    labels : list
        Input labels
    class_names : list
        List of class names. The order of names should correspond to class index at classifier output.
    format_out : str
        Wanted output format. one of {'names', 'indices'}

    Returns
    -------
    labels : list
        Output labels in wanted format
    """

    # get indices <-> names mapping
    name_to_ind, ind_to_name = class_names_indices_mapping(class_names)

    if format_out == 'names':
        if (not isinstance(labels[0], str)):  # if labels are indices, convert to strings
            labels = [ind_to_name[ind] for ind in labels]

    if format_out == 'indices':
        if (not np.issubdtype(type(labels[0]), np.integer)):  # if labels are indices, convert to strings
            labels = [name_to_ind[ind] for ind in labels]

    return labels


def compute_colors_for_labels(labels, color_factor=1):
    """
    Simple function that adds fixed colors depending on the class

    Parameters
    ----------
    labels : array-like
        Array of class indices.
    color_factor : float, optional
        Parameter which scale the colors.

    Returns
    -------
    colors : ndarray
        Colors for each label.
    """

    labels = np.asarray(labels) * color_factor

    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    colors = labels[:, None] * palette
    colors = (colors % 255).astype("uint8")

    return colors


def overlay_boxes(image, boxes, class_names=None, color_factor=1, thickness=2, image_size_factor=1):
    """
    Overlay bounding boxes on top of image.

    Parameters
    ----------
    image : ndarray
        Input image.
    boxes : Box
        Box instance. Should include lables
    class_names : list, optional
        List of class names. The order of names should correspond to class index at classifier output.
    color_factor : float, optional
        Parameter which scale the colors.
    thickness : int, optional
        Parameter which controls box thickness.
    image_size_factor : float, optional
        Scale displayed parameter (e.g. text, lines) by image size - for nicer display.

    Returns
    -------
    image : ndarray
        Output image.
    """

    # scale thickness
    thickness *= image_size_factor
    thickness = np.maximum(int(np.around(thickness)), 1)  # cast to int, verify that thicness is at least 1

    # get bounding boxes
    bboxes = boxes.bbox

    # get labels - for bboxes colors
    if boxes.has_field('labels'):  # use labels if exist
        labels = boxes.get_field('labels')

        # verify that labels are numbers
        if (len(labels) > 0) and isinstance(labels[0], str) and (class_names is not None):  # if labels are strings, convert to indices
            name_to_ind = class_names_indices_mapping(class_names)[0]
            labels = [name_to_ind[name] for name in labels]

    else:  # assume all labels are ones
        labels = np.ones((len(boxes))).tolist()

    # set boxes colors
    colors = compute_colors_for_labels(labels, color_factor=color_factor).tolist()

    # overlay boxes
    for box, color in zip(bboxes, colors):
        box = box.astype(np.int32)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), thickness)

    return image, colors


def overlay_scores_and_class_names(image, boxes, class_names=None, colors=None, text_position='above', text_size_factor=1, image_size_factor=1):
    """
    Overlay class names on top of image.

    Parameters
    ----------
    image : ndarray
        Input image.
    boxes : Box
        Box instance. Should include labels
    class_names : list, optional
        List of class names. The order of names should correspond to class index at classifier output.
    colors : list, optional
        List of colors.
    text_position : str, optional
        Sets position of class names above or below boubding box, one of {'above', 'below'}.
    text_size_factor : float, optional
        Sets font size.
    image_size_factor : float, optional
        Scale displayed paramenter (e.g. text, lines) by image size - for nicer display.

    Returns
    -------
    image : ndarray
        Output image.
    """

    text_size_factor *= image_size_factor
    text_thickness = 2 * image_size_factor
    text_thickness = np.maximum(int(np.around(text_thickness)), 1)  # cast to int, verify that thicness is at least 1

    try:  # if there are prediction scores
        scores = boxes.get_field("scores").tolist()
        template = "{}: {:.2f}"  # label: score
    except:
        scores = ['' for _ in range(len(boxes))]
        template = "{}{}"  # label

    # get labels
    labels = boxes.get_field("labels")

    # verify that labels are strings, convert them if necessary
    if class_names is not None:
        labels = convert_labels(labels, class_names, format_out='names')

    # get bounding boxes
    bboxes = boxes.bbox

    # set text parameters
    font_size = 1. * text_size_factor  # font size
    if colors is None:
        colors = ((255, 255, 255), ) * len(labels)

    # overlay class names and scores
    for box, score, label, color in zip(bboxes, scores, labels, colors):

        # set text
        text = template.format(label, score)

        # get text size
        (text_width, text_height), text_baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)  # get text size

        # set text bottom left corner
        text_left = int(0.5 * (box[0] + box[2] - text_width))
        text_right = int(0.5 * (box[0] + box[2] + text_width))
        if text_position == 'above':
            text_top = int(box[1]- text_height)
            text_bottom = int(box[1])
        elif text_position == 'below':
            text_top = int(box[3])
            text_bottom = int(box[3] + text_height)
        text_bottom_left_corner = (text_left, text_bottom)
        top_left = (text_left, text_top)
        bottom_right = (text_right, text_bottom)

        # set color
        # option 1
        text_color = color
        background_color = (int(255 - text_color[0]), int(255 - text_color[1]), int(255 - text_color[2]))
        # # option 2
        # background_color = color
        # text_color = (int(255 - background_color[0]), int(255 - background_color[1]), int(255 - background_color[2]))
        # # option 3
        # text_color = color
        # background_color = (255, 255, 255)


        # overlay text
        image = cv2.rectangle(image, top_left, bottom_right, background_color, -1)  # overlay background
        cv2.putText(image, text, text_bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, text_thickness)  # overlay text

    return image


def convert_pyplot_figure_to_ndarray(fig, dpi=180):
    """
    Convert matplotlib.pyplot figure to numpy.ndarray.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        Figure handle.
    dpi : int
        Wanted resolution in dots per inch

    Returns
    -------
    img : ndarray
        Output image.
    """

    # create buffer
    buf = io.BytesIO()

    # save figure to buffer
    fig.savefig(buf, format='png', dpi=dpi)

    # convert to ndarray
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def rotate_xlabels(fig, ax, bottom=0.2, rotation=30, ha='right', which=None):
    """
    Based on matplotlib.figure.autofmt_xdate().

    Date ticklabels often overlap, so it is useful to rotate them
    and right align them.  Also, a common use case is a number of
    subplots with shared xaxes where the x-axis is date data.  The
    ticklabels are often long, and it helps to rotate them on the
    bottom subplot and turn them off on other subplots, as well as
    turn off xlabels.

    Parameters
    ----------
    fig : matplotlib.figure
        Figure object.

    ax : matplotlib.Axes
     Axes object.

    bottom : scalar
        The bottom of the subplots for :meth:`subplots_adjust`.

    rotation : angle in degrees
        The rotation of the xtick labels.

    ha : str
        The horizontal alignment of the xticklabels.

    which : {None, 'major', 'minor', 'both'}
        Selects which ticklabels to rotate. Default is None which works
        the same as major.
    """

    # set rotation for selected axis
    for label in ax.get_xticklabels(which=which):
        label.set_ha(ha)
        label.set_rotation(rotation)

    # adjust size
    fig.tight_layout()
    # if ax.is_last_row():
    #     fig.subplots_adjust(bottom=bottom, left=bottom)

class VideoProcessor(object):
    """
    Simple class for writing videos.
    """

    def __init__(self, fps=12,  output_video_path='video.avi'):
        """
        Initialize VideoProcessor object.

        Parameters
        ----------
        fps : int, optional
            Video frame rate in frames per second.
        output_video_path : str, optional
            Output video path.

        Returns
        -------
        None.
        """
        self.fps = fps
        self.output_video_path = output_video_path
        self.fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        self.length = 0

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "fps={}, ".format(self.fps)
        s += "output_video_path={}".format(self.output_video_path)
        s += "length={}".format(self.length)
        return s

    def __len__(self):
        return self.length


    def write_frames(self, frame_list):
        """
        Write frames to video.

        Parameters
        ----------
        frame_list : list
            List of frames to be written in video.

        Returns
        -------
        None.
        """

        # get frame width and height
        height, width = frame_list[0].shape[:2]

        # create video writer object
        video_writer = cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (width, height))

        # write frames to video
        counter = 0
        for frame in frame_list:
            video_writer.write(frame)
            counter += 1

        self.length = counter

        # close video writer
        video_writer.release()