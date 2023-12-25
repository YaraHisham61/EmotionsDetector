import cv2 as cv2
import numpy as np
from skimage import io
from commonFunctions import *


class SkinDetector:
    def __init__(self,img_path) :
        """
        Initializes a Skin Detector with the given image.

        Parameters:
        - img (np.array): image to detect the skin from it.
        """
        self.img=io.imread(img_path,as_gray=True)
        self.skOut =cv2.imread(img_path)


    def detect (self) :
        """
        Detects the skin from the image.

        Returns:
        - np.array: image with the skin detected.
        """
        
        #converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(self.skOut, cv2.COLOR_BGR2HSV)
        #skin color range for hsv color space 
        HSV_mask = cv2.inRange(img_HSV, (0, 48, 80), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(self.skOut, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 140, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result=cv2.bitwise_not(global_mask)

        contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or max(cv2.contourArea(c) for c in contours) < (60 * 60):
            self.img = self.img
        
        else:
            for contour in contours: 
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.skOut, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around external contour

            is_square_like = lambda cnt: abs(cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] - 1) < 0.5

                # Filter contours based on the square-like condition
            filtered_contours = [cnt for cnt in contours if is_square_like(cnt)]
            if(len(filtered_contours) == 0 or max(cv2.contourArea(c) for c in filtered_contours) < (60 * 60) ):
                external_contour = max(contours, key=cv2.contourArea)
            else:
                # Find the contour with maximum area among filtered contours
                external_contour = max(filtered_contours, key=cv2.contourArea)

                # Extract bounding box
            x, y, w, h = cv2.boundingRect(external_contour)
            cv2.rectangle(self.skOut, (x, y), (x + w, y + h), (0,0, 255), 10)
                # Crop the image
            
            self.img = self.img[y:y + h, x:x + w]

        cv2.imwrite ("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/skin_detect_out.jpg", self.skOut)
        cv2.imwrite ("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/1_HSV.jpg",HSV_result)
        cv2.imwrite ("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/2_YCbCr.jpg",YCrCb_result)
        cv2.imwrite ("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/3_global_result.jpg",global_result)

        # self.img = (self.img * 255).astype(np.uint8)
        # self.img = image_resize(self.img, width = 48, height = 48, inter = cv2.INTER_AREA)
        self.img = cv2.resize(self.img, (48, 48), interpolation = cv2.INTER_AREA)

    
        print(self.img.shape)
        io.imsave ("E:/Collage/IP/Project/EmotionDetector/EmotionsDetector/SkinDetection/haar_cascade_in.png", (self.img * 255).astype(np.uint8))

        


if __name__ == "__main__":
    sk = SkinDetector("SkinDetection/20211217_212902.jpg")
    sk.detect()
