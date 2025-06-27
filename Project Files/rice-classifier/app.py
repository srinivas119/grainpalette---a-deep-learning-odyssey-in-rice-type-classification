from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/rice_model.h5')
labels = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    file = request.files.get('file')

    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = labels[np.argmax(pred)]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
