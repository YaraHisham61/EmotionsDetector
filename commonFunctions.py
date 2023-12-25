import numpy as np
from PIL import Image
import os
from HaarCascade.haarLikeFeatures import HaarLikeFeature, FeatureType
from functools import partial
import cv2

def load_images(path, target_size=(48, 48)):

    """
    Loads images from a directory and converts them to numpy arrays.

    Parameters:
    
    - path (str): Path to the directory containing the images.
    - target_size (tuple[int]): Target size of the images after resizing.

    Returns:

    - List[numpy.ndarray]: List of numpy arrays of the images.
    """
    images = []
    for _file in os.listdir(path):
        img = Image.open(os.path.join(path, _file))
        img_resized = img.resize(target_size)
        img_arr = np.array(img_resized, dtype=np.float64) / 255.0  # Normalize to [0, 1]
        images.append(img_arr)
    return images

def load_image(path, target_size=(48, 48)):

    """"
    Loads an image from a file path and converts it to a numpy array.

    Parameters:

    - path (str): Path to the image file.
    - target_size (tuple[int]): Target size of the image after resizing.

    Returns:

    - numpy.ndarray: Numpy array of the image.
    """
    img = Image.open(os.path.join(path))
    # Convert to grayscale
    img_gray = img.convert('L')
        
    # Resize the grayscale image
    img_resized = img_gray.resize(target_size)
    img_arr = np.array(img_resized, dtype=np.float64) / 255.0  # Normalize to [0, 1]
    return [img_arr]

def ensemble_vote(int_img, classifiers):
    """
    Performs an ensemble vote on a given integral image using a set of Haar-like feature classifiers.

    Parameters:
    - int_img (numpy.ndarray): Integral image.
    - classifiers (List[HaarLikeFeature]): List of Haar-like feature classifiers.

    Returns:
    - int: The ensemble vote result, where 1 indicates a positive classification and 0 indicates a negative classification.
    """
    classifier_votes_sum = sum(c.get_vote(int_img) for c in classifiers)
    return 1 if classifier_votes_sum >= 0 else 0



def ensemble_vote_all(int_imgs, classifiers):

    """
    Applies an ensemble vote to a list of integral images using a set of Haar-like feature classifiers.

    Parameters:
    - int_imgs (List[numpy.ndarray]): List of integral images to be classified.
    - classifiers (List[HaarLikeFeature]): List of Haar-like feature classifiers.

    Returns:
    - List[int]: List of assigned labels, where 1 indicates a positive classification and 0 indicates a negative classification.
    """

    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized