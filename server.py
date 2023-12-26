from flask import Flask, jsonify
from flask_cors import CORS  # Import the CORS module
from flask import request
from PIL import Image
import HaarCascade.integralImage as int_img
from SkinDetection.skinDetector import SkinDetector
from commonFunctions import *
from skimage import io
from joblib import load
from main import load_classifiers

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['SAVE_PATH'] = 'my_trained_classifiers_sd.pkl'
app.config['CLF'] = load('EmotionsModel/rbf.joblib')
app.config['CLASSIFIERS'] = load_classifiers(app.config['SAVE_PATH'])
app.config['PHOTO_PATH'] = 'D:/Collage/IP/EmotionsDetector/SkinDetection/imgR.jpg'
app.config['OUT_PATH'] = 'D:/Collage/IP/EmotionsDetector/SkinDetection/haar_cascade_in.png'

@app.route('/')
def hello_world():
    json_file = {}
    json_file['query'] = 'hello_world'
    return jsonify(json_file)

@app.route('/image', methods=['POST'])
def image():
    file = request.files['image']
    img = Image.open(file.stream)
    img.save(app.config['PHOTO_PATH'])

    sd = SkinDetector(app.config['PHOTO_PATH'])
    sd.detect()

    print('Loading pre-trained classifiers...')
    

    image_ii = int_img.calc_integral_image(sd.img)

    result = [ensemble_vote(image_ii,classifiers=app.config['CLASSIFIERS'])]
    if(result[0] == 0):
        print("not face")
        result = [3]
    else:
        print("face")
        img = io.imread(app.config['OUT_PATH'])
        img = img.flatten()
        result = app.config['CLF'].predict([img])

    return jsonify({
                'result': str(result[0]) 
           })


if __name__ == '__main__':
    app.run(debug=True)
