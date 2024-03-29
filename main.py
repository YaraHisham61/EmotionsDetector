import HaarCascade.integralImage as int_img
from SkinDetection.skinDetector import SkinDetector
from commonFunctions import *
import pickle
from joblib import load
from skimage import io


def load_classifiers(file_path):
    with open(file_path, 'rb') as file:
        classifiers = pickle.load(file)
    return classifiers

def get_img_path ():
    return 'SkinDetection/mic.jpg'
     

if __name__ == "__main__":
    #skin detect first
    sd = SkinDetector(get_img_path())
    sd.detect()

    classifiers = []
    save_path = 'my_trained_classifiers_sd.pkl'

    print('Loading pre-trained classifiers...')
    classifiers = load_classifiers(save_path)

    image_ii = int_img.calc_integral_image(sd.img)

    result = ensemble_vote(image_ii,classifiers=classifiers)
    if(result == 0):
        print("not face")
    else:
        print("face")
        clf = load('EmotionsModel/rbf.joblib')
        testImage = io.imread("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/haar_cascade_in.png")
        testImage = testImage.flatten()
        p = clf.predict([testImage])
        print(p)






