import numpy as np
from HaarCascade.haarLikeFeatures import HaarLikeFeature, FeatureType

class adaBoost : 
    def __init__(self,positive_images,negative_images) -> None:
        """
        Initializes an AdaBoost classifier.

        Parameters:
        - positive_images (List[numpy.ndarray]): List of positive images.
        - negative_images (List[numpy.ndarray]): List of negative images.
        """

        self.positive_images = positive_images
        self.negative_images = negative_images
        self.classifiers = []
        self.num_classifiers = 3

    def train (self):
        """
        Trains the AdaBoost classifier.

        Returns:
        - List[HaarLikeFeature]: List of Haar-like features.
        """
        num_pos = len(self.positive_images)
        num_neg = len(self.negative_images)
        num_imgs = num_pos + num_neg

        pos_weights = np.ones(num_pos) *1. / (2 * num_pos)
        neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
        self.weights = np.hstack((pos_weights, neg_weights))

        labels = np.hstack((np.ones(num_pos), -1 * np.ones(num_neg)))
        # Initialize classifiers
        img_height = min(self.positive_images[0].shape[0] , self.negative_images[0].shape[0])
        img_width = min(self.positive_images[0].shape[1] , self.negative_images[0].shape[1])

        for _ in range(self.num_classifiers):
            # Create Haar-like features and select the best feature
            features = self.create_features(img_height, img_width, 1,3,1, 3)

            feature, feature_weight, self.weights = self.select_best_feature(self.positive_images + self.negative_images, self.weights, labels, features)

            # Store the selected feature and its weight in self.classifiers
            self.classifiers.append(feature)

        return self.classifiers


    def create_features(self, img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
        """
        Creates Haar-like features of different sizes and types.

        Parameters:
        - img_height (int): Height of the images.
        - img_width (int): Width of the images.
        - min_feature_width (int): Minimum width of the features.
        - max_feature_width (int): Maximum width of the features.
        - min_feature_height (int): Minimum height of the features.
        - max_feature_height (int): Maximum height of the features.

        Returns:
        - List[HaarLikeFeature]: List of Haar-like features.
        """
        features = []
        total_iterations = 100000
        for feature_type in FeatureType:
            feature_start_width = max(min_feature_width, feature_type.value[0])
            for feature_width in range(feature_start_width, max_feature_width, feature_type.value[0]):
                feature_start_height = max(min_feature_height, feature_type.value[1])
                for feature_height in range(feature_start_height, max_feature_height, feature_type.value[1]):
                    for x in range(0, img_width - feature_width):
                        for y in range(0, img_height - feature_height):
                            features.append(HaarLikeFeature(feature_type, x, y, feature_width, feature_height, 0, 1))
                            features.append(HaarLikeFeature(feature_type, x, y, feature_width, feature_height, 0, -1))

        return features


    def select_best_feature(self,images, weights, labels, features):
        """
        Selects the best Haar-like feature at each iteration of AdaBoost.

        Parameters:
        - images (List[numpy.ndarray]): List of integral images.
        - weights (numpy.ndarray): Array of weights for each image.
        - labels (numpy.ndarray): Array of labels for each image (1 for positive, -1 for negative).
        - features (List[HaarLikeFeature]): List of Haar-like features.

        Returns:
        - HaarLikeFeature: The best-selected Haar-like feature.
        - float: Weight assigned to the selected feature.
        - numpy.ndarray: Updated weights for each image.
        """

        num_images = len(images)
        num_features = len(features)
        self.weights *= 1. / np.sum(self.weights )

        # Initialize arrays to store the votes and errors for each feature
        votes = np.zeros((num_images, num_features))
        errors = np.zeros(num_features)

        # Calculate the votes for each feature on each image
        for i in range(num_images):
            votes[i, :] = np.array([feature.get_vote(images[i]) for feature in features])

        # Iterate over each feature to find the best one
        for f in range(num_features):
            # Calculate the error for the current feature
            errors[f] = sum(weights * (labels != votes[:, f]))

        # Find the index of the feature with the minimum error
        best_feature_index = np.argmin(errors)
        best_feature = features[best_feature_index]

        # Calculate the weight assigned to the best feature
        error = errors[best_feature_index]
        if error > 0 and error < 1:
            feature_weight = 0.5 * np.log((1 - error) / error)
        else:
            # Handle the case where error is exactly 0 or 1
            feature_weight = 0

        # Update the weights for each image
        weights *= np.exp(-feature_weight * labels * votes[:, best_feature_index])
        weights /= np.sum(weights)

        return best_feature, feature_weight, weights
