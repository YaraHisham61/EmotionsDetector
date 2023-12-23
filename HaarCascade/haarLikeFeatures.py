import numpy as np
import HaarCascade.integralImage as int_img
from enum import Enum

class FeatureType(Enum):
    TWO_VERTICAL = (1, 2)
    TWO_HORIZONTAL = (2, 1)
    THREE_VERTICAL = (1, 3)
    THREE_HORIZONTAL = (3, 1)
    FOUR_DIAGONAL = (2, 2)

FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR_DIAGONAL]

class HaarLikeFeature:
    def __init__(self, feature_type,x0,y0,w,h,threshold,polarity):
        """
        Initializes a Haar-like feature with the given type and position.

        Parameters:
        - feature_type (FeatureType): Type of the feature.
        - x0 (int): Starting column of the rectangle.
        - y0 (int): Starting row of the rectangle.
        - w (int): Width of the rectangle.
        - h (int): Height of the rectangle.
        - threshold (int): Threshold of the feature.
        - polarity (int): Polarity of the feature.
        """
        self.type = feature_type
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h

        # The threshold represents a numerical value that is used to compare against the computed score of a Haar-like feature. This score is obtained by applying the feature to a particular region of the image.
        # The decision based on the threshold is typically binary: if the computed score is above the threshold, the region is classified as positive (contains an object), and if it's below the threshold, the region is classified as negative (does not contain an object).
        self.threshold = threshold

        # The polarity defines the direction of the feature's response in relation to the threshold. It can take values of -1 or 1.
        # If the polarity is 1, a region is classified as positive (object present) when the feature's score is less than the threshold.
        # If the polarity is -1, a region is classified as positive when the feature's score is greater than the threshold.
        self.polarity = polarity
        self.weight = 1

    def get_score(self, integral_image):
        """
        Calculates the score of the feature on the given integral image.

        Parameters:
        - integral_image (numpy.ndarray): Integral image of size (h+1, w+1).

        Returns:
        int: Score of the feature on the given integral image.
        """
        white =0 
        black =0

        if self.type == FeatureType.TWO_VERTICAL:
            white = int_img.calc_region_sum(integral_image, self.x0, self.y0, self.w, self.h / 2)
            black = int_img.calc_region_sum(integral_image, self.x0, self.y0 + self.h / 2, self.w, self.h / 2)

        elif self.type == FeatureType.TWO_HORIZONTAL:
            white = int_img.calc_region_sum(integral_image, self.x0, self.y0, self.w / 2, self.h)
            black = int_img.calc_region_sum(integral_image, self.x0 + self.w / 2, self.y0, self.w / 2, self.h)

        elif self.type == FeatureType.THREE_VERTICAL:
            white = int_img.calc_region_sum(integral_image, self.x0, self.y0, self.w, self.h/3) + int_img.calc_region_sum(integral_image, self.x0, self.y0+ self.h*2/3, self.w, self.h/3)
            black = int_img.calc_region_sum(integral_image, self.x0, self.y0 + self.h/3, self.w, self.h/3)
        
        elif self.type == FeatureType.THREE_HORIZONTAL:
            white = int_img.calc_region_sum(integral_image, self.x0, self.y0, self.w/3, self.h) + int_img.calc_region_sum(integral_image, self.x0 + self.w*2/3, self.y0, self.w/3, self.h)
            black = int_img.calc_region_sum(integral_image, self.x0 + self.w/3, self.y0, self.w/3, self.h)
        
        elif self.type == FeatureType.FOUR_DIAGONAL:
            white = int_img.calc_region_sum(integral_image, self.x0, self.y0, self.w/2, self.h/2) + int_img.calc_region_sum(integral_image, self.x0 + self.w/2, self.y0 + self.h/2, self.w/2, self.h/2)
            black = int_img.calc_region_sum(integral_image, self.x0 + self.w/2, self.y0, self.w/2, self.h/2) + int_img.calc_region_sum(integral_image, self.x0, self.y0 + self.h/2, self.w/2, self.h/2)
        
        return white - black

    def get_vote(self, integral_image):
        """
        Calculates the vote of the feature on the given integral image.

        Parameters:
        - integral_image (numpy.ndarray): Integral image of size (h+1, w+1).

        Returns:
        int: Vote of the feature on the given integral image.
        """

        score= self.get_score(integral_image)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)
