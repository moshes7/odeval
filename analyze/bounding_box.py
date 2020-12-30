import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
import cv2

class Box(object):
    """
    Class for working with bounding boxes.
    Inspired by maskrcnn_benchmark.structures.bounding_box.BoxList().
    """

    def __init__(self, bbox, image_shape, bbox_type='ltrb', extra_fields=None):
        """
        Initialize Box object.

        Parameters
        ----------
        bbox : ndarray
            Array of shape (N, 4), each row represent bounding box.
        image_shape : tuple
            Shape of image to which bounding boxes belong, given as (height, width).
        bbox_type : str, optional
            Type of bounding box representation. one of {'ltrb', }.
            - 'ltrb': each row of bbox is of the form [left, top, right, bottom]
        extra_fields: dict or None, optional
            If not None, should be a dict with additional wanted values, such as classification score, label names etc.

        Returns
        -------
        Box object.
        """
        # cast to ndarray
        bbox = np.asarray(bbox, dtype=np.float32)

        # verify correctness
        if (bbox.size > 0) and (bbox.ndim != 2):  # bbox must be either empty or of shape (N, 4)
            raise ValueError("bbox should have 2 dimensions, got {}".format(bbox.ndimension()))

        if (bbox.size > 0) and (bbox.shape[-1] != 4):
            raise ValueError("last dimension of bbox should have a size of 4, got {}".format(bbox.size(-1)))

        if (not isinstance(image_shape, tuple)) or ((len(image_shape) != 2) and (len(image_shape) != 3)):
            raise ValueError("image_shape must be tuple of length 2 (height, width) or 3 (height, width, channels), got {}".format(image_shape))

        if bbox_type not in ['ltrb']:
            raise ValueError("mode should be 'ltrb'")

        self.bbox = bbox
        self.image_shape = image_shape  # (height, width)
        self.bbox_type = bbox_type
        self.extra_fields = {}

        if extra_fields is not None:
            for key, val in extra_fields.items():
                self.add_field(key, val)

    def add_field(self, field, field_data):
        """
        Add field to existing Box object.

        Parameters
        ----------
        field : str
            Field name.
        field_data : any data type
            Field data.

        Returns
        -------
        None
        """
        self.extra_fields[field] = field_data

    def get_field(self, field):
        """
        Get field to existing Box object.

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        Field data.
        """
        return self.extra_fields[field]

    def has_field(self, field):
        """
        Check if Box has certain field.

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        True if field exist, False otherwise.
        """
        return field in self.extra_fields

    def __getitem__(self, key):

        # cast key to list
        if not isinstance(key, list) and not isinstance(key, slice) and not isinstance(key, np.ndarray):
            key = [key]

        # slice using key
        bboxes = self.bbox[key, :]  # shape (len(key), 4)
        if bboxes.ndim == 1:  # verify shape of (:, 4)
            bboxes = bboxes[np.newaxis, :]

        extra_fields = {}
        for k, v in self.extra_fields.items():
            extra_fields[k] = [v[ind] for ind in key]  # type is list
            if isinstance(v, np.ndarray):  # cast type to ndarray
                extra_fields[k] = np.asarray(extra_fields[k])

        bbox = Box(bbox=bboxes,
                   image_shape=self.image_shape,
                   bbox_type=self.bbox_type,
                   extra_fields=extra_fields)

        return bbox

    def __delitem__(self, key):

        # delete bbox data
        self.bbox = np.delete(self.bbox, (key), axis=0)  # delete row

        # delete extra fields data
        for key_ef, val_ef in self.extra_fields.items():
            if isinstance(val_ef, np.ndarray):
                self.extra_fields[key_ef] = np.delete(val_ef, (key))  # delete row
            if isinstance(val_ef, list):
                val_ef.__delitem__(key)

    def __len__(self):
        return self.bbox.shape[0]

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self):
            raise StopIteration
        else:
            return self.__getitem__(self._index)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.image_shape[1])
        s += "image_height={}, ".format(self.image_shape[0])
        s += "bbox_type={})".format(self.bbox_type)
        return s


    def insert(self, ind, box):
        """
        Insert bbox and extra_fields in box into current instance at index ind.
        Assume that all attributes (image_shape, bbox_type, etc.) are identical.

        Parameters
        ----------
        ind : int
            Index in which bbox will be inserted.
        box : bounding_box.Box
            Bounding box to be inserted.

        Returns
        -------
        None
        """

        # verity bbox type
        if not isinstance(box, Box):
            raise ValueError("bbox type shold be 'Box'")

        # treat case of ind == -1
        ind_insert = ind if ind != -1 else self.bbox.shape[0]  # since -1 inserts at the BEGGINING, not end

        # insert values to bbox
        self.bbox = np.insert(self.bbox, ind_insert, box.bbox, axis=0)

        # insert values to extra_fields
        for key, val_list in box.extra_fields.items():

            if isinstance(val_list, list):  # treat list
                ind_current = ind_insert
                for val in val_list:
                    self.extra_fields[key].insert(ind_current, val)
                    ind_current += 1

            elif isinstance(val_list, np.ndarray):  # treat ndarray
                self.extra_fields[key] = np.insert(self.extra_fields[key], ind_insert, val_list, axis=0)


    def append(self, box):
        """
        Append bbox and extra_fields in box into current instance.
        Assume that all attributes (image_shape, bbox_type, etc.) are identical.

        Parameters
        ----------
        box : bounding_box.Box
            Bounding box to be appended.

        Returns
        -------
        None
        """
        self.insert(-1, box)

    @staticmethod
    def concatenate(box_seq):
        """
        Concatenate boxes sequence to one box.
        Attributes (e.g. image_shape, bbox_type) will be taken from first elemnt in sequence.

        Parameters
        ----------
        box_seq : Sequence (e.g. list, tuple)
            Sequence of bounding box to be concatenated.

        Returns
        -------
        box: bounding_box.Box()
            Concatenated box.
        """
        if len(box_seq) > 0:
            box = Box.copy(box_seq[0])  # initialize with first sequence element
            for n in range(1, len(box_seq)):  # append all other elements
                box.append(box_seq[n])
        else:
            box = box_seq

        return box

    @staticmethod
    def copy(box):
        """
        Make a copy of box.

        Parameters
        ----------
        box : bounding_box.Box
            Input box.

        Returns
        -------
        box_out : bounding_box.Box
             Box copy.
        """

        box_out = Box(bbox=box.bbox, image_shape=box.image_shape, bbox_type=box.bbox_type, extra_fields=box.extra_fields)
        return box_out


    def area(self):
        """
        Calculate area of all boxes in current object, in pixels.

        Returns
        -------
        area : ndarray
            Array of areas.
        """

        bbox = self.bbox
        area = Box.calculate_bbox_area(bbox, bbox_type=self.bbox_type)

        return area


    @staticmethod
    def calculate_bbox_area(bbox, bbox_type):
        """
        Calculate boxes area in pixels.

        Parameters
        ----------
        bbox : ndarray
            Bounding boxes array.
        bbox_type : str, optional
            Type of bounding box representation. one of {'ltrb', }.
            - 'ltrb': each row of bbox is of the form [left, top, right, bottom]

        Returns
        -------
        area : ndarray
            Array of areas.
        """

        if bbox_type == "ltrb":
            width = bbox[:, 2] - bbox[:, 0] + 1  # add 1 to include edge pixel
            height = bbox[:, 3] - bbox[:, 1] + 1  # add 1 to include edge pixel
            area = width * height
        else:
            raise RuntimeError("Unsupported bbox type!")

        return area


    @staticmethod
    def change_bounding_box_type(bbox_in, type_in, type_out):
        """
        Change bounding box type.

        Parameters
        ----------
        bbox_in : array-like
            Array defining a bounding box, according to type_in.

        type_in, type_out : str
            Types of input and output bounding boxes, one of {tl_br, cvat_polygon}:

            ltrb:
                array of 4 numbers defining the top left and bottom right corners coordinates:
                    [x_left, y_top, x_right, y_bottom].

            cvat_polygon:
                array of shape (5, 2) of the following form:
                    [[x_left, y_top],
                     [x_right, y_top],
                     [x_right, y_bottom],
                     [x_left, y_bottom],
                     [x_left, y_top]].

            corners:
                array of shape (4, 2) of the following form:
                    [[x_left, y_top],
                     [x_right, y_top],
                     [x_right, y_bottom],
                     [x_left, y_bottom]].

        Returns
        -------
        bbox_out : ndarray
            Array defining a bounding box, according to type_out.
        """

        # get input bounding box parameters
        if type_in == 'ltrb':
            left = bbox_in[0]
            top = bbox_in[1]
            right = bbox_in[2]
            bottom = bbox_in[3]

        else:
            left = bbox_in[:, 0].min()
            top = bbox_in[:, 1].min()
            right = bbox_in[:, 0].max()
            bottom = bbox_in[:, 1].max()

        # calculate output bounding box
        if type_out == 'ltrb':

            bbox_out = np.array([left, top, right, bottom])

        elif type_out == 'cvat_polygon':

            bbox_out = np.array([[left, top],
                                 [right, top],
                                 [right, bottom],
                                 [left, bottom],
                                 [left, top]])

        elif type_out == 'corners':

            bbox_out = np.array([[left, top],
                                 [right, top],
                                 [right, bottom],
                                 [left, bottom]])

        # verify dtype float32
        bbox_out = bbox_out.astype(np.float32)

        return bbox_out


    @ staticmethod
    def boxes_iou(boxes1, boxes2):
        """
        Compute the intersection over union of two set of boxes.
        The box order must be (left, top, right, bottom).

        Parameters
        ----------
        boxes1 : Box or ndarray
            Box object of length N.
        boxes2 : Box or ndarray
            Box object of length M.

        Returns
        -------
        iou : ndarray
            IOU array of shape (N, M).

        References
        ----------
            https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
            https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
        """

        if isinstance(boxes1, Box):  # boxes are Box
            # verify that images has same shapes
            # if boxes1.image_shape != boxes2.image_shape:
            #     raise RuntimeError("boxlists should have same image shape, got {}, {}".format(boxes1.image_shape, boxes2.image_shape))

            # get bounding boxes
            bbox1 = boxes1.bbox  # (N,4)
            bbox2 = boxes2.bbox  # (M,4)

            # calculate area
            area1 = boxes1.area()
            area2 = boxes2.area()

        else:  # boxes are ndarray

            # get bounding boxes
            bbox1 = boxes1  # (N,4)
            bbox2 = boxes2  # (M,4)

            # calculate area
            area1 = Box.calculate_bbox_area(bbox1, bbox_type=boxes1.bbox_type)
            area2 = Box.calculate_bbox_area(bbox2, bbox_type=boxes2.bbox_type)

        # calculate intersection area
        lt = np.maximum(bbox1[:, None, :2], bbox2[:, :2])  # (N,M,2) left top coordinates
        rb = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])  # (N,M,2) right bottom coordinates

        wh = np.clip(rb - lt + 1, a_min=0, a_max=None)  # (N,M,2) add 1 to include edge pixel
        intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        # calculate union area
        union = area1[:, None] + area2 - intersection  # [:, None] implicitly adds new axis for broadcasting

        # calculate iou
        iou = intersection / union

        return iou


    @staticmethod
    def bbox_centers(bbox, bbox_type='ltrb', x_weight=0.5, y_weight=0.5):
        """
        Calculate boxes weighted center, using the following formula:
            centers_x = x_weight * left + (1 - x_weight) * right
            centers_y = y_weight * top + (1 - y_weight) * bottom

        Parameters
        ----------
        bbox : Box or ndarray
            Bounding boxes array, of shape (N, 4).
        bbox_type : str, optional
            Type of bounding box representation. one of {'ltrb', }.
        x_weight : float, optional
            Weight of left-right center.
        y_weight : float, optional
            Weight of top-bottom center.

        Returns
        -------
        centers : list
            list of N weighted (x, y) centers.
        """

        if isinstance(bbox, Box):  # boxes are Box
            bbox = bbox.bbox  # (N,4)

        # convert bbox type to ltrb
        if bbox_type != 'ltrb':
            bbox = Box.change_bounding_box_type(bbox, bbox_type, 'ltrb')

        # calculate bbox centers
        centers = np.array([x_weight * (bbox[:, 0]) + (1 - x_weight) * bbox[:, 2],
                            y_weight * (bbox[:, 1]) + (1 - y_weight) * bbox[:, 3]]).T.tolist()

        return centers

    @staticmethod
    def is_centers_inside_bbox(bbox, bbox2centers, bbox_type='ltrb', x_weight=0.5, y_weight=0.5):
        """
        check if bbox2centers (weighted) centers are inside bbox.

        Parameters
        ----------
        bbox : Box or ndarray
            Box object of length N, in which we check if centers are inside.
        bbox2centers : Box or ndarray
            Box object of length M, from which centers will be calculated.
        bbox_type : str, optional
            Type of bounding box representation. one of {'ltrb', }.
        x_weight : float, optional
            Weight of left-right center.
        y_weight : float, optional
            Weight of top-bottom center.

        Returns
        -------
        is_center_inside : ndarray
            is_center_inside array of shape (N, M).
            Entry (m, n) is equal to 1 if center m is inside bbox n, and 0 otherwise.
         """

        if isinstance(bbox, Box):  # boxes are Box
            # get bounding boxes array
            bbox = bbox.bbox  # (N, 4)

        if isinstance(bbox2centers, Box):  # boxes are Box
            # get bounding boxes array
            bbox2centers = bbox2centers.bbox  # (M, 4)

        # convert bbox type to ltrb
        if bbox_type != 'ltrb':
            bbox = Box.change_bounding_box_type(bbox, bbox_type, 'ltrb')
            bbox2centers = Box.change_bounding_box_type(bbox2centers, bbox_type, 'ltrb')

        # verify that bbox have 2 dims
        if bbox.ndim < 2:
            bbox = bbox[np.newaxis, :]

        if bbox2centers.ndim < 2:
            bbox2centers = bbox2centers[np.newaxis, :]

        # calculate centers
        centers = Box.bbox_centers(bbox2centers, x_weight=x_weight, y_weight=y_weight)
        centers = [Point(center[0], center[1]) for center in centers]  # convert centers to shapely Point() objects

        # check if centers are inside boxes
        is_center_inside = np.zeros((bbox.shape[0], bbox2centers.shape[0]), dtype=np.int)  # shape (N, M)
        for n in range(bbox.shape[0]):

            # convert bounding box to polygon
            polygon = Box.bounding_box_2_polygon(bbox[n, :])

            # check if centers are inside polygon
            is_inside = [polygon.contains(center) for center in centers]

            # convert to ndarray
            is_inside = np.array(is_inside)  # shape (M, )

            # save current vector as the n'th row of is_center_inside
            is_center_inside[n, is_inside] = 1

        return is_center_inside

    @staticmethod
    def bounding_box_2_polygon(bbox, bbox_type='ltrb'):
        """
        Create Polygon from bounding box.

        The bounding box is first converted to the following standard form

            [top_left, top_right, bottom_right, bottom_left, top_left]

        or more explicitly

            [[x_left, y_top],
             [x_right, y_top],
             [x_right, y_bottom],
             [x_left, y_bottom],
             [x_left, y_top]]

        Than, the standard bounding box is converted to polygon.

        Parameters
        ----------
        bbox : array-like
            Array defining a bounding box, according to type.

        bbox_type : str, optional
            Type of bounding box, one of {ltrb, }:

        Returns
        -------
        polygon : shapely.geometry.polygon.Polygon
            Polygon of the bounding box.
        """

        # convert bbox type to ltrb
        if bbox_type != 'ltrb':
            bbox = Box.change_bounding_box_type(bbox, bbox_type, 'ltrb')

        # extract bounding box parameters
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        # set bounding box standard form:
        # traverse boundary clock-wise starting - and ending - at the top-left corner
        bbox_standard_form = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]  # closed line

        # define polygon
        polygon = Polygon(bbox_standard_form)

        return polygon

    def resize(self, image_shape_output):
        """
        Resize boxes according to output image shape.

        Parameters
        ----------
        output_image_shape : tuple
            Image shape (height, width).

        Returns
        -------
        None
        """

        # calculate transformation between original and resized images
        corners_orig = Box.get_corners_rect(self.image_shape).astype(np.float32)[:3]  # corner coordinates of original image
        corners_resized = Box.get_corners_rect(image_shape_output).astype(np.float32)[:3]# corner coordinates of resized image
        M = cv2.getAffineTransform(corners_orig, corners_resized)

        # warp bounding box
        bbox = self.bbox.reshape((-1, 2))
        bbox_resized = Box.get_warped_points(bbox, M)
        # bbox_resized = np.around(bbox_resized).astype(np.int)  # cast to int
        bbox_resized = bbox_resized.reshape((-1, 4))

        self.bbox = bbox_resized
        self.image_shape = image_shape_output

        return

    @ staticmethod
    def get_corners_rect(shape, TLCorner=(0, 0)):
        """
        Get corners coordinates of rectangle given its' shape and top left corner coordinates.

        Parameters
        ----------

        TLCorner : tuple
            Coordinates of rectangle top left corner, given in (x,y) = (col,row).

        shape: tuple
            rectangle shape.

        Returns
        ----------
        corners: 4x2 ndarray
            Coordinates of rectangle corners ordered as follows:
            [topLeft, topRight, bottomRight, bottomLeft]
            The coordinates of each point are given (x,y), i.e (col,row).
        """

        h = shape[0]
        w = shape[1]

        corners = TLCorner + np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        return corners

    @staticmethod
    def get_warped_points(points, M):

        """
        Apply affine transformation to 2D points and calculate resulting coordinates.

        Parameters
        ----------
        points: ndarray, float/double
            Coordinates of points to be transformed, of shape (N, 2).
            Each point is given by 2D row vector ordered as (x,y) i.e. (col, row).

        M: 2x3 matrix
            Affine transformation matrix.

        Returns
        ----------
        points_warped: ndarray
            Coordinates of transformed points, of shape (N, 2).
            Each point is given by 2D row vector ordered as (x,y) i.e. (col,row)).
        """

        # verify that points type is ndarray, convert if not
        if type(points).__module__ != np.__name__:
            points = np.array(points)[np.newaxis, :]

        # reverse order of input points [y,x] -> [x,y]
        # points = points[...,::-1]

        # Find full affine marix
        rowM = np.array([[0, 0, 1]])
        M = np.concatenate((M, rowM), axis=0)

        size = len(points.shape)
        # p=points.copy() # for debug

        # option 1 - use cv2.perspectiveTransform()
        # cv2.perspectiveTransform() expects to receive 3D array, so we need to verify that points has 3 dimensions
        for m in range(3 - size):
            points = points[np.newaxis, ...]

        points_warped = cv2.perspectiveTransform(points.astype(np.float64), M)
        points_warped = np.squeeze(points_warped)

        # reverse order of input points [y,x] -> [x,y]
        # points_warped = points_warped[..., ::-1]

        '''
        # option 2 - use matrix multiplication
        # assumes points are ordered (y,x)!
        points = p # for debug
        if size == 1:
            rowP = np.ones(1)
        elif size == 2:
            rowP = np.ones(N)

        points = np.concatenate((points, rowP), axis=0)
        points_warped2 = np.dot(M, points)
        points_warped2 = points_warped2[:-1]

        diff = np.sum(np.abs(points_warped2 - points_warped)) # for debug
        '''
        return points_warped