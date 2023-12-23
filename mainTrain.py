from HaarCascade.adaBoost import adaBoost
from commonFunctions import *
import HaarCascade.integralImage as int_img

import pickle

def train_and_save_classifiers(pos_training_path, neg_training_path, save_path):
      print('Loading faces..')
      faces_training = load_images(pos_training_path)
      faces_ii_training = list(map(int_img.calc_integral_image, faces_training))
      print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
      non_faces_training = load_images(neg_training_path)
      non_faces_ii_training = list(map(int_img.calc_integral_image, non_faces_training))
      print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

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
      pos_training_path = 'E:/Collage/IP/Project/EmotionDetector/positiveImages'
      neg_training_path = 'E:/Collage/IP/Project/EmotionDetector/negativeImages'
      pos_testing_path = 'E:/Collage/IP/Project/EmotionDetector/testPositiveImages'
      neg_testing_path = 'E:/Collage/IP/Project/EmotionDetector/testNegativeImages'
      classifiers = []
      save_path = 'my_trained_classifiers_1_10.pkl'

      # Check if the trained classifiers are already saved
      if not os.path.exists(save_path):
            train_and_save_classifiers(pos_training_path, neg_training_path, save_path)
      else:
            print('Loading pre-trained classifiers...')
            classifiers = load_classifiers(save_path)
      
      print('Loading test faces..')
      faces_testing = load_images(pos_testing_path)
      faces_ii_testing = list(map(int_img.calc_integral_image, faces_testing))
      print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
      non_faces_testing = load_images(neg_testing_path)
      non_faces_ii_testing = list(map(int_img.calc_integral_image, non_faces_testing))
      print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

      print('Testing selected classifiers..')
      correct_faces = 0
      correct_non_faces = 0
      correct_faces = sum(ensemble_vote_all(faces_ii_testing, classifiers))
      correct_non_faces = len(non_faces_testing) - sum(ensemble_vote_all(non_faces_ii_testing, classifiers))

      print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
            + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
            + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
            + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

