import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
# import utils.preprocessing.image as image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import nasnet
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import json
import os
import imghdr
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--class_index', default='index_to_class.json', type = str, help = 'index map')
parser.add_argument('--valdir', default='test', type = str, help = 'input dir')
parser.add_argument('--input', default='input.csv', type = str, help = 'input images')
args = parser.parse_args()

print(args.class_index)
print(args.input)

start = time.time()

labels = [line.rstrip('\n') for line in open(args.class_index)]
print(labels)

models224 = []
models224.append(keras.models.load_model("output/best2/densenet_169-20190311-111348/finetune/model.h5"))
models224.append(tf.contrib.saved_model.load_keras_model("output/best/mobilenet_v2-20190302-114111/finetune/savedmodel/1551706666"))

models299 = []
models299.append(keras.models.load_model("output/best/xception-20190224-090007/finetune/model.h5"))

acc = 0
top3 = 0
top5 = 0
total = 0

data = pd.read_csv(args.input, names = ['id', 'label']) 
for index, row in data.iterrows():
    filename = args.valdir + "/" + row['id'] + '.jpg'

    if os.path.exists(filename) == 0: # removed files 
        print("Not found: ", filename)
        continue 
    if os.path.getsize(filename) == 0: # zero-byte files 
        os.remove(filename)
        print("Zero: ", filename)
        continue 
    if imghdr.what(filename) not in ['jpeg', 'png', 'gif']: # invalid image files
        os.remove(filename)
        print("Invalid: ", filename)
        continue
        
    truthLabel = row['label']

    imgs224 = image.load_img(filename, target_size=(224,224))
    x224 = image.img_to_array(imgs224)
    x224 = np.expand_dims(x224, axis=0)
    x224 = preprocess_input(x224)

    preds = []
    for model in models224:
        preds.append(model.predict(x224))

    imgs299 = image.load_img(filename, target_size=(299,299))
    x299 = image.img_to_array(imgs299)
    x299 = np.expand_dims(x299, axis=0)
    x299 = preprocess_input(x299)

    for model in models299:
        preds.append(model.predict(x299))

    pred = np.mean(preds, axis=0)[0]

    top = pred.argsort()[-5:][::-1]
    top_labels = list(map(lambda x: labels[x], top))
    top_confidents = list(map(lambda x: pred[x], top))

    total+=1
    if truthLabel in top_labels[:3] :
        top3+=1
    else :
        print('[', row['id'], '] Predicted: ', top_labels, ', Confident=', top_confidents, ", truth=", truthLabel)
        mid_time = time.time() - start
        print('Progress after ',mid_time,': acc=', acc, ', top3=', top3, ', top5=', top5, ' / total=', total)
        sys.stdout.flush()

    if truthLabel in top_labels[:5] :
        top5+=1
    
    if truthLabel == top_labels[0] :
        acc+=1    
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

exec_time = time.time() - start
print('Finished after ',exec_time,': acc=', acc, ', top3=', top3, ', top5=', top5, ' / total=', total)