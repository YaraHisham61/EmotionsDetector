# Emotion Detector App
## <img align= center width=30px height=30px src="https://github.com/AhmedSamy02/Adders-Mania/assets/88517271/dba75e61-02dd-465b-bc31-90907f36c93a"> Table of Contents

- [Overview](#overview)
- [System Description](#sysdes) 
- [Used Algorithms](#algo)
- [Experiment results](#res)
- [Demo Video](#vid)
- [How To Run](#run)
- [Used Libraries](#lib)
- [Contributors](#contributors)
- [License](#license)
  
## <img src="https://github.com/AhmedSamy02/Adders-Mania/assets/88517271/9ed3ee67-0407-4c82-9e29-4faa76d1ac44" width="30" height="30" /> Overview <a name = "overview"></a>
This app is based on taking or picking up photos and detecting which it's a face or not and if it's a face it classifies which emotion is [Happy - Sad - Surprised]
## <img src="https://github.com/YaraHisham61/OS_Scheduler/assets/88517271/d8e6c9f3-9ba5-4d9e-a108-7d9a95989812" width="30" height="30" /> System Description <a name = "sysdes"></a>
![image](https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/1931d295-de99-4bf1-8e8e-e7ac992e1459)
## <img src="https://cdn-icons-png.flaticon.com/512/7690/7690595.png" width="30" height="30" /> Used Algorithms <a name = "algo"></a>
- Skin Detection
  - Skin detection algorithms typically involve color thresholding in specific color spaces, We combined HSV and YCbCr
  results. The algorithm identifies pixels with color values within a predefined range that corresponds to skin tones. 
- Haar Cascade
  - The Haar Cascade algorithm is a machine learning-based object detection method used to identify objects or 
  features in images. It employs a cascade of simple classifiers trained on positive and negative samples. The 
  algorithm evaluates image regions at multiple scales and resolutions, using Haar-like features to distinguish 
  between the object and the background. Training involves adjusting weights for these features. While widely 
  used for face detection, Haar Cascade can be trained for various objects, making it versatile in applications 
  like security surveillance and robotics.
- SVM (RBF)
  - Support Vector Machines (SVM) with Radial Basis Function (RBF) kernel is a popular machine learning 
  algorithm used for classification and regression tasks. The RBF kernel allows SVM to handle non-linear 
  relationships between input features. The SVM with RBF kernel is particularly effective in capturing complex 
  patterns and is widely used in image recognition due to its ability to handle diverse and non-linear data.
## <img src="https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/6726d0d6-d0e6-4b7d-942a-3e03430ccbb8" width="30" height="30" /> Experiment results <a name = "res"></a>
- Skin Detection
  - As it is a color thresholding algorithm, relying solely on color information, it is not highly 
  accurate in distinguishing skin tones from similar colors found in non-human skin-like objects, 
  such as beaches or yellowish items.
  - To detect the human face, we must get the biggest contour, and has to be square-like.
- Haar Cascade 
  - Different size of images was used as input to the algorithm. We discovered that the less 
  resolution the image is the more accurate the classification is.
  - Trying different widths and heights of features and different numbers of classifiers, we found that 
  the best number of classifiers is 3 and the dimensions of features are to be limited to (10 x 10) to 
  get the best results in a reasonable training time with:
    - Training set of 26k positive images and 13k negative images 
    - 3 Strong classifiers 
    - 2000 photos for face test 
    - 500 photos for non-faces test.
    - Faces accuracy: 75% 
    - Non-faces accuracy: 50 %
- Model Training
  - At first, We started discarding some model trainers such as SVM by Sigmoid as we needed more than 
  two classes while the sigmoid function classifies between only 2 classes. We started as first step with data 
  preprocessing or data cleaning. Then we started with 7 classes to classify but some emotions may cause conflicts with others so we tried to decrease the number of classes to increase the 
  validation accuracy the emotions that conflict like sad and disgust` decrease the accuracy of the model we chose only three emotions (Happy – Sad – Surprised ) and we trained using SVM with 
  RBF kernel why this kernel because as mentioned it allows SVM to handle non-linear relationships 
  and of course, the emotions are non-linear We tried other model trainers but it was in vain because the 
  accuracy was very horrible.
  One of the challenges that faced us were low accuracy as the total accuracy was 66.89% which can 
  be divided into: 
    - Happy accuracy = 80.38%
    - Sad accuracy = 59.74%
    - Surprised accuracy = 60.53%
## <img src="https://github.com/YaraHisham61/AYKN-Search-Engine/assets/88517271/2783fa4c-1371-45d2-bbfa-7682bbc4b65d" width="30" height="30" /> Demo Video <a name = "vid"></a>


https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/11a0f3ca-5785-49ee-9efb-4784f59c7739


## <img src="https://github.com/YaraHisham61/OS_Scheduler/assets/88517271/1c40c081-3619-449b-a9d7-605fc7b2baca" width="30" height="30" />  How To Run <a name = "run"></a>
1) Clone the project
```
git clone https://github.com/YaraHisham61/EmotionsDetector
 ```
2) Go to the project directory
  ```
  cd EmotionsDetector
  ```
3) Open the server
```
 python server.py  
 ```
4) Go to the GUI directory
```
  cd emotion_detector
  ```
5) Get the dependencies
  ```
  flutter pub get
  ``` 
6) Run the code
```
  flutter run
  ```
7) Upload the photo using the GUI and wait for the output

## <img src="https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/e309ac2c-bc8f-409d-a444-45ac2272086a" width="20" height="20" /> Used Libraries  <a name = "lib"></a>
### Python <img src="https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/26018fd6-5ad9-4759-a033-59f6a9502bb7" width="30" height="30" />
- [OpenCV](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html)
- [Sckit-learn](https://scikit-learn.org/stable/modules/classes.html)
- [Joblib](https://joblib.readthedocs.io/en/stable/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [Numpy](https://numpy.org/)
- [Pillow](https://pypi.org/project/Pillow/)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)

### Flutter <img src="https://github.com/YaraHisham61/EmotionsDetector/assets/88517271/838951f0-3acb-4c51-89ab-3ab9969f8e16" width="30" height="30" />
- [Dio](https://pub.dev/packages/dio)
- [Animated Emoji](https://pub.dartlang.org/packages?q=animated_emoji)
- [Page Transition](https://pub.dartlang.org/packages?q=page_transition)
- [Image Picker](https://pub.dartlang.org/packages?q=image_picker)
- [Flutter Svg](https://pub.dartlang.org/packages?q=flutter_svg) 
- [Flutter Animate](https://pub.dartlang.org/packages?q=flutter_animate)

## <img src="https://github.com/YaraHisham61/OS_Scheduler/assets/88517271/859c6d0a-d951-4135-b420-6ca35c403803" width="30" height="30" /> Contributors <a name = "contributors"></a>
<table>
  <tr>
   <td align="center">
    <a href="https://github.com/AhmedSamy02" target="_black">
    <img src="https://avatars.githubusercontent.com/u/96637750?v=4" width="150px;" alt="Ahmed Samy"/>
    <br />
    <sub><b>Ahmed Samy</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/kaokab33" target="_black">
    <img src="https://avatars.githubusercontent.com/u/93781327?v=4" width="150px;" alt="Kareem Samy"/>
    <br />
    <sub><b>Kareem Samy</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/nancyalgazzar" target="_black">
    <img src="https://avatars.githubusercontent.com/u/94644017?v=4" width="150px;" alt="Nancy Ayman"/>
    <br/>
    <sub><b>Nancy Ayman</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/YaraHisham61" target="_black">
    <img src="https://avatars.githubusercontent.com/u/88517271?v=4" width="150px;" alt="Yara Hisham"/>
    <br />
    <sub><b>Yara Hisham</b></sub></a>
    </td>
  </tr>
 </table>
 
 ## <img src="https://github.com/YaraHisham61/Architecture_Project/assets/88517271/c4a8b264-bf74-4f14-ba2a-b017ef999151" width="30" height="30" /> License <a name = "license"></a>
> This software is licensed under MIT License, See [License](https://github.com/YaraHisham61/EmotionsDetector/blob/master/LICENSE)
