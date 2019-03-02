import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
# import utils.preprocessing.image as image
from tensorflow.keras.applications.inception_v3 import preprocess_input
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
parser.add_argument('--model', default='model.h5', type = str, help = 'model in h5 format')
parser.add_argument('--size', default=299, type = int, help = 'img size')
args = parser.parse_args()

print(args.class_index)
print(args.input)
print(args.model)

start = time.time()
# index_to_class = {}
# with open(args.class_index) as json_file:  
#     index_to_class = json.load(json_file)

# print(index_to_class)

labels = [line.rstrip('\n') for line in open(args.class_index)]
print(labels)

model = keras.models.load_model(args.model)

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

    # imgs = image.load_img_crop(filename, target_size=(args.size, args.size))
    # x = image.imgs_to_array(imgs)
    imgs = image.load_img(filename, target_size=(args.size, args.size))
    x = image.img_to_array(imgs)
    # plt.figure(figsize=(15,100))
    # for i in range(6):
    #     plt.subplot(1,6,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(imgs[i], cmap=plt.cm.binary)
    # plt.show()
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    pred = np.mean(preds, axis=0)
    # for pred in preds:
    # print(pred)

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