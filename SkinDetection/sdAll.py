import cv2
import numpy as np
import os

input_directory = 'E:/Collage/IP/ProjectMaterial/ls'
output_directory = 'E:/Collage/IP/Project/EmotionDetector/newOut'

# Loop through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more file extensions if needed
        # Read the image
        img = cv2.imread(os.path.join(input_directory, filename))
        #converting from gbr to hsv color space

        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #skin color range for hsv color space 
        HSV_mask = cv2.inRange(img_HSV, (0, 48, 80), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 140, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

        # Find contours in the global_mask
        contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) == 0 or max(cv2.contourArea(c) for c in contours) < (60 * 60)):
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            destination_path = os.path.join(output_directory, filename)
            cv2.imwrite(destination_path, gray_img)

            continue
        
        is_square_like = lambda cnt: abs(cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] - 1) < 0.5

        # Filter contours based on the square-like condition
        filtered_contours = [cnt for cnt in contours if is_square_like(cnt)]
        
        
        if(len(filtered_contours) == 0 or max(cv2.contourArea(c) for c in filtered_contours) < (60 * 60) ):
            external_contour = max(contours, key=cv2.contourArea) 
        else :
        # Find the contour with maximum area among filtered contours
            
            external_contour = max(filtered_contours, key=cv2.contourArea)

        # Extract bounding box
        x, y, w, h = cv2.boundingRect(external_contour)
        cropped_img = img[y:y + h, x:x + w]
        # resized_img = cv2.resize(cropped_img, (48, 48))
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        destination_path = os.path.join(output_directory, filename)

        cv2.imwrite(destination_path, gray_img)
