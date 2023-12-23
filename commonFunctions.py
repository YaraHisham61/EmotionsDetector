import numpy as np
from PIL import Image
import os
from HaarCascade.haarLikeFeatures import HaarLikeFeature, FeatureType
from functools import partial

def load_images(path, target_size=(48, 48)):
    images = []
    for _file in os.listdir(path):
        img = Image.open(os.path.join(path, _file))
        img_resized = img.resize(target_size)
        img_arr = np.array(img_resized, dtype=np.float64) / 255.0  # Normalize to [0, 1]
        images.append(img_arr)
    return images

def load_image(path, target_size=(48, 48)):
    img = Image.open(os.path.join(path))
    # Convert to grayscale
    img_gray = img.convert('L')
        
        # Resize the grayscale image
    img_resized = img_gray.resize(target_size)
    img_arr = np.array(img_resized, dtype=np.float64) / 255.0  # Normalize to [0, 1]
    return [img_arr]

def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


