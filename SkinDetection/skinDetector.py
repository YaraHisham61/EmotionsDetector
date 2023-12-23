import cv2 as cv2
import numpy as np

class SkinDetector:
    def __init__(self,img_path) :
        """
        Initializes a Skin Detector with the given image.

        Parameters:
        - img (np.array): image to detect the skin from it.
        """
        self.img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.skOut =cv2.imread(img_path)


    def detect (self) :
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
        if len(contours) > 0:
            for contour in contours: 
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.skOut, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around external contour

            is_square_like = lambda cnt: abs(cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] - 1) < 0.5

            # Filter contours based on the square-like condition
            filtered_contours = [cnt for cnt in contours if is_square_like(cnt)]

            # Find the contour with maximum area among filtered contours
            external_contour = max(filtered_contours, key=cv2.contourArea)

            # Extract bounding box
            x, y, w, h = cv2.boundingRect(external_contour)

            # Crop the image
            self.img = self.img[y:y + h, x:x + w]

        cv2.imwrite ("SkinDetection/haar_cascade_in.jpg", self.img)
        cv2.imwrite ("SkinDetection/skin_detect_out.jpg", self.skOut)
        cv2.imwrite ("SkinDetection/1_HSV.jpg",HSV_result)
        cv2.imwrite ("SkinDetection/2_YCbCr.jpg",YCrCb_result)
        cv2.imwrite ("SkinDetection/3_global_result.jpg",global_result)

        # self.img = cv2.resize(self.img, (48, 48))
        self.img = np.array(self.img, dtype=np.float64) / 255.0



if __name__ == "__main__":
    sk = SkinDetector("SkinDetection/20211217_212902.jpg")
    sk.detect()
