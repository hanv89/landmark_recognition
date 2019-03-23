#Usage: python app.py
import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import models
import numpy as np
import argparse
# import imutils
# import cv2
import time
import uuid
import base64
import json

img_width, img_height = 299, 299
model_path = './zaloai_models/inception_v3-remodel-20190121.h5'
model_map = './my_inception_v3/index_to_class.json'
model = models.load_model(model_path)
model._make_predict_function() # have to initialize before threading

index_to_class = {}
with open(model_map) as json_file:  
    index_to_class = json.load(json_file)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    img = image.load_img(file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    top = preds[0].argsort()[-3:][::-1]
    top = list(map(lambda x: index_to_class[str(x)], top))

    print("Labels: ", top)
    return top

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            # result = ['1','2','3']
            label = filename + ' = ' + ' > '.join(result)
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)