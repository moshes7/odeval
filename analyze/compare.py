import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compare_analyzer_dirs(dir1, dir2, output_dir, concat_type='horizontal', sfx_output='png', resize_factor=1, generate_video=True):
    """
    Compare 2 analyzer directories.

    Parameters
    ----------
    dir1 : str
        First directory.
    dir2 : str
        Second directory.
    output_dir : str
        Output directory.
    concat_type : str, optional
        Concatenation type, one of {'horizontal', 'vertical'}.
    sfx_output : str, optional
        Suffix (file type) of output images.
    generate_video : bool, optional
        If True, a video will be generated in output directory.

    Returns
    -------
    None.
    """

    # compare images
    images_dir1 = os.path.join(dir1, 'images')
    images_dir2 = os.path.join(dir2, 'images')
    compare_images_in_two_dirs(images_dir1, images_dir2, output_dir, concat_type=concat_type, sfx_output=sfx_output,
                               resize_factor=resize_factor, generate_video=generate_video)

    # compare metrics
    compare_images_in_two_dirs(dir1, dir2, output_dir, concat_type=concat_type, sfx_output=sfx_output,
                               resize_factor=1, generate_video=False)

    return



def compare_images_in_two_dirs(dir1, dir2, output_dir, concat_type='horizontal', sfx_output='png', resize_factor=1, generate_video=True):
    """
    Compare corresponding images from 2 differect directories by concatenating them to 1 image and saving the result in
    output_dir.

    Parameters
    ----------
    dir1 : str
        First directory.
    dir2 : str
        Second directory.
    output_dir : str
        Output directory.
    concat_type : str, optional
        Concatenation type, one of {'horizontal', 'vertical'}.
    sfx_output : str, optional
        Suffix (file type) of output images.
    generate_video : bool, optional
        If True, a video will be generated in output directory.

    Returns
    -------
    None.
    """

    # verify that output dir exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # get image numbers and paths in dict in the format {image_number: image_full_path}
    images_dict_1 = {'{}'.format(int(image_name.split('.')[0])) : os.path.abspath(os.path.join(dir1, image_name))
                     for image_name in os.listdir(dir1)}
    images_dict_2 = {'{}'.format(int(image_name.split('.')[0])) : os.path.abspath(os.path.join(dir2, image_name))
                     for image_name in os.listdir(dir2)}

    # find common keys
    keys1 = np.fromiter(images_dict_1.keys(), dtype=np.int)
    keys2 = np.fromiter(images_dict_2.keys(), dtype=np.int)
    keys = np.intersect1d(keys1, keys2)
    keys = np.sort(keys)

    # generate output images directory
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)

    # iterate over keys
    for n, key in tqdm(enumerate(keys)):

        # read images
        img1 = cv2.imread(images_dict_1[str(key)], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(images_dict_2[str(key)], cv2.IMREAD_UNCHANGED)

        # resize
        if resize_factor != 1:
            img1 = cv2.resize(img1, None, None, fx=resize_factor, fy=resize_factor)
            img2 = cv2.resize(img2, None, None, fx=resize_factor, fy=resize_factor)

        # concatenate
        img_concat = concatenate_images(img1, img2, concat_type=concat_type)

        # save concatenate image
        img_name = os.path.join(output_images_dir, '{:06d}.{}'.format(key, sfx_output))
        cv2.imwrite(img_name, img_concat)

    if generate_video:
        output_file = os.path.join(output_dir, 'video.mp4')
        generate_video_from_images(output_images_dir, output_file, img_sfx=sfx_output)

    return

def concatenate_images(left, right, concat_type='horizontal'):
    """
    Concatenate 2 images to 1 image.

    Parameters
    ----------
    left : ndarray
        Left image.
    right : ndarray
        Right image.
    concat_type : str, optional
        Concatenation type, one of {'horizontal', 'vertical'}. Default is 'horizontal'.
        If 'vertial', left is top and right is bottom.

    Returns
    -------
    image_concat : ndarray
        Output image.
    """

    # select concatenation axis
    if concat_type == 'horizontal':
        axis = 1
    elif concat_type == 'vertical':
        axis = 0

    # verify that images hase the same size, adjust sizes if neccesary
    left, right = match_image_size(left, right)

    # concatenate images
    img_concat = np.concatenate((left, right), axis=axis)

    return img_concat

def match_image_size(img1, img2):
    """
    Match images size such that both images will have the same size, by padding zeros to the smaller image.

    Parameters
    ----------
    img1 : ndarray
        First input image.
    img2 : ndarray
        Secong input image.

    Returns
    -------
    img1_padd : ndarray
        First output image.
    img2_pad : ndarray
        Secong output image.
    """

    # get images shape
    h1, w1, c = img1.shape  # assume that both images have the same number of color channels c
    h2, w2, c = img2.shape

    # get maximum values
    h_max = np.maximum(h1, h2)
    w_max = np.maximum(w1, w2)

    # pad images as needed
    if (h1 < h_max) or (w1 < w_max):
        img1_pad = np.zeros((h_max, w_max, c), dtype=img1.dtype)
        img1_pad[0:h1, 0:w1] = img1
    else:
        img1_pad = img1

    if (h2 < h_max) or (w2 < w_max):
        img2_pad = np.zeros((h_max, w_max, c), dtype=img2.dtype)
        img2_pad[0:h2, 0:w2] = img2
    else:
        img2_pad = img2

    return img1_pad, img2_pad

def generate_video_from_images(images_dir, output_file, img_sfx='jpg', fps=10, max_frames_num=None):
    """
    Generate video file from images diretory.

    Parameters
    ----------
    images_dir : str
        Images directory.
    output_file : str
        Output video file name.
    img_sfx : str, optional
        Images suffix.
    fps : int, optional
        Video frame rate.
    max_frames_num : int, optional
        If not None, sets the maximum number of images from which the video will be created.

    Returns
    -------
    None.
    """
    # get image list
    image_list = [os.path.join(images_dir, file_name) for file_name in os.listdir(images_dir)] #if file_name.endswith(img_sfx)]

    if max_frames_num is not None:
        image_list = image_list[:max_frames_num]

    N = len(image_list)

    # get image shape
    img = cv2.imread(image_list[0], cv2.IMREAD_UNCHANGED)
    try:
        h, w, c = img.shape
    except:
        h, w = img.shape
        c = 0

    # instantiate video object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    # write images to video
    for n, img_name in tqdm(enumerate(image_list), desc='generating video - {}'.format(os.path.basename(images_dir))):
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    return
