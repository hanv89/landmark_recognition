import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
# import utils.preprocessing.image as image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import nasnet
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import json
import os
import imghdr
import sys
import time
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--class_index', default='index_to_class.json', type = str, help = 'index map')
parser.add_argument('--valdir', default='test', type = str, help = 'input dir')
parser.add_argument('--output', default='output.csv', type = str, help = 'output images')
parser.add_argument('--model', default='model.h5', type = str, help = 'model in h5 format')
parser.add_argument('--savedmodel', type = str, help = 'savedmodel dir')
parser.add_argument('--size', default=299, type = int, help = 'img size')
args = parser.parse_args()

print(args.class_index)
print(args.output)
print(args.model)
print(args.savedmodel)

start = time.time()

labels = [line.rstrip('\n') for line in open(args.class_index)]
print(labels)

if not args.savedmodel:
    model = keras.models.load_model(args.model)
else:
    model = tf.contrib.saved_model.load_keras_model(args.savedmodel)

total = 0

results = dict()
uris = pathlib.Path(args.valdir).glob('**/*.jpg')
for uri in uris:
    filepath, f = os.path.split(uri)
    filename, ext = os.path.splitext(f)

    if os.path.exists(uri) == 0: # removed files 
        print("Not found: ", uri)
        continue 
    if os.path.getsize(uri) == 0: # zero-byte files 
        os.remove(uri)
        print("Zero: ", uri)
        continue 
    if imghdr.what(uri) not in ['jpeg', 'png', 'gif']: # invalid image files
        os.remove(uri)
        print("Invalid: ", uri)
        continue
        
    imgs = image.load_img(uri, target_size=(args.size, args.size))
    x = image.img_to_array(imgs)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    pred = np.mean(preds, axis=0)

    top = pred.argsort()[-5:][::-1]
    top_labels = list(map(lambda x: labels[x], top))
    top_confidents = list(map(lambda x: pred[x], top))

    results[filename] = ' '.join(top_labels[:3])
    total+=1
    print('total:', total,', file:', filename, ':', results[filename])
    sys.stdout.flush()
    
with open(args.output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','predicted'])
    for result in results.keys():
        writer.writerow([result, results[result]])

exec_time = time.time() - start
print('Finished after ',exec_time,' total=', total)