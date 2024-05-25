from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from skimage import feature
import cv2
import pickle

app = Flask(__name__)

def quantify_image(image):
    features = feature.hog(image, orientations=9, pixels_per_cell=(10, 10), 
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return features

with open("ParkinsonsDrawingPrediction (1).pkl", "rb") as file:
    model = pickle.load(file)

def predict_parkinson_percentage(image):
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (200, 200))
    image_thresholded = cv2.threshold(image_resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image_thresholded)
    probs = model.predict_proba([features])[0]
    parkinson_percentage = 100 - (probs[0] * 100)
    return parkinson_percentage

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['file']

    try:
        image = Image.open(file.stream)
        score = predict_parkinson_percentage(image)

        response = {'prediction': str(score), "message": "predicted successfully",'status':200}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e),'status':404}), 500

if __name__ == '__main__':
    print("server is working")
    app.run(debug=False)
