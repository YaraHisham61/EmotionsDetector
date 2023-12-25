from HaarCascade.adaBoost import adaBoost
from commonFunctions import *
import HaarCascade.integralImage as int_img

import pickle

def train_and_save_classifiers(pos_training_path, neg_training_path, save_path):
      """
      Trains the AdaBoost classifier and saves the trained classifiers to a file.

      Parameters:
      - pos_training_path (str): Path to the directory containing the positive training images.
      - neg_training_path (str): Path to the directory containing the negative training images.
      - save_path (str): Path to the file to save the trained classifiers to.
      """
      
      print('Loading training faces...')
      # Load training face images
      faces_training = load_images(pos_training_path)
      # Calculate integral images for the training face images
      faces_ii_training = list(map(int_img.calc_integral_image, faces_training))
      print(f'Done. {len(faces_training)} faces loaded.\n')

      print('Loading training non-faces...')
      # Load training non-face images
      non_faces_training = load_images(neg_training_path)
      # Calculate integral images for the training non-face images
      non_faces_ii_training = list(map(int_img.calc_integral_image, non_faces_training))
      print(f'Done. {len(non_faces_training)} non-faces loaded.\n')


      ab = adaBoost (faces_ii_training, non_faces_ii_training)
      # classifiers are haar-like features
      classifiers = ab.train ()

      # Save the trained classifiers
      with open(save_path, 'wb') as file:
            pickle.dump(classifiers, file)

      print('Classifiers saved to:', save_path)

def load_classifiers(file_path):


      with open(file_path, 'rb') as file:
            classifiers = pickle.load(file)
      return classifiers

if __name__ == "__main__":
      pos_training_path = 'E:/Collage/IP/Project/EmotionDetector/newFaces'
      neg_training_path = 'E:/Collage/IP/Project/EmotionDetector/newOut'
      pos_testing_path = 'E:/Collage/IP/Project/EmotionDetector/testPositiveImages'
      neg_testing_path = 'E:/Collage/IP/Project/EmotionDetector/testNegativeImages'
      classifiers = []
      save_path = 'my_trained_classifiers_sd.pkl'

      # Check if the trained classifiers are already saved
      if not os.path.exists(save_path):
            train_and_save_classifiers(pos_training_path, neg_training_path, save_path)
      else:
            print('Loading pre-trained classifiers...')
            classifiers = load_classifiers(save_path)
      
      print('Loading test faces...')
      # Load test face images
      faces_testing = load_images(pos_testing_path)
      # Calculate integral images for the test face images
      faces_ii_testing = list(map(int_img.calc_integral_image, faces_testing))
      print(f'Done. {len(faces_testing)} faces loaded.\n')

      print('Loading test non-faces...')
      # Load test non-face images
      non_faces_testing = load_images(neg_testing_path)
      # Calculate integral images for the test non-face images
      non_faces_ii_testing = list(map(int_img.calc_integral_image, non_faces_testing))
      print(f'Done. {len(non_faces_testing)} non-faces loaded.\n')

      print('Testing selected classifiers...')
      # Initialize counters for correct classifications
      correct_faces = sum(ensemble_vote_all(faces_ii_testing, classifiers))
      correct_non_faces = len(non_faces_testing) - sum(ensemble_vote_all(non_faces_ii_testing, classifiers))

      print('Done.\n')
      # Display the results
      print(f'Result:\n'
            f'    Faces: {correct_faces}/{len(faces_testing)}  '
            f'({(correct_faces / len(faces_testing)) * 100:.2f}%)\n'
            f'non-Faces: {correct_non_faces}/{len(non_faces_testing)}  '
            f'({(correct_non_faces / len(non_faces_testing)) * 100:.2f}%)')


