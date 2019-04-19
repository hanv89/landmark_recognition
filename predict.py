import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import utils.utils as utils
import random
import json
import argparse
import glob
import time
import math
import sys
import os


parser = argparse.ArgumentParser()
parser.add_argument('--labels', default='labels/label_name.csv', type = str, help = 'list of labels')
parser.add_argument('--input', default='image.jpg', type = str, help = 'input image or image dir')
parser.add_argument('--model', default='model.h5', type = str, help = 'model in h5 format')
parser.add_argument('--savedmodel', type = str, help = 'savedmodel dir')
parser.add_argument('--size', default=224, type = int)
args = parser.parse_args()

if os.path.exists(args.input) and os.path.exists(args.model) and os.path.exists(args.labels):
    label_names = [line.rstrip('\n') for line in open(args.labels)]
    print(label_names)

    if not args.savedmodel:
        model = keras.models.load_model(args.model)
    else:
        model = tf.contrib.saved_model.load_keras_model(args.savedmodel)

    imgs = []
    uris = [args.input]
    if os.path.isdir(args.input):
        uris = glob.glob(args.input + '/*.jpg')
    
    for uri in uris:
        print(uri)
        img = image.load_img(uri, target_size=(args.size,args.size))
        img = image.img_to_array(img)
        imgs.append(img)
    
    x = preprocess_input(np.array(imgs))
    preds = model.predict(x, batch_size=2, verbose=1)

    tops = []
    tops_labels = []
    tops_confidents = []
    for pred in preds:
        top = pred.argsort()[-5:][::-1]
        tops.append(top)
        top_labels = list(map(lambda x: label_names[x], top))
        tops_labels.append(top_labels)
        top_confidents = list(map(lambda x: pred[x], top))
        tops_confidents.append(top_confidents)

    # show

    # plt.figure(figsize=(100,100))
    for i in range (0,len(imgs)):
        plt.subplot(4,4,i%16+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.array_to_img(imgs[i]))
        plt.xlabel(tops_labels[i][0] + ":" + "{:10.2f}".format(tops_confidents[i][0]))
        if (i+1)%16==0:
            plt.show()
    if not len(imgs)%16 == 0:
        plt.show()
else:
    print("No such file or dir", args.input, args.model, args.labels)



