import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import json
import os
import imghdr

parser = argparse.ArgumentParser()
parser.add_argument('--class_index', default='index_to_class.json', type = str, help = 'index map')
parser.add_argument('--valdir', default='test', type = str, help = 'input dir')
parser.add_argument('--input', default='input.csv', type = str, help = 'input images')
parser.add_argument('--model', default='model.h5', type = str, help = 'model in h5 format')
args = parser.parse_args()

print(args.class_index)
print(args.input)
print(args.model)

index_to_class = {}
with open(args.class_index) as json_file:  
    index_to_class = json.load(json_file)

print(index_to_class)

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

    img = image.load_img(filename, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    top = preds[0].argsort()[-5:][::-1]
    top_labels = list(map(lambda x: index_to_class[str(x)], top))
    top_confidents = list(map(lambda x: preds[0][x], top))

    total+=1
    if truthLabel in top_labels[:3] :
        top3+=1

    if truthLabel in top_labels[:5] :
        top5+=1
    
    if truthLabel == top_labels[0] :
        acc+=1    

    # plt.figure(figsize=(10,10))
    # # plt.xticks([])
    # # plt.yticks([])
    # plt.imshow(x[0])
    # # plt.imshow(img, cmap=plt.cm.binary)
    # plt.colorbar()
    # plt.grid(False)
    # plt.xlabel(top)
    # plt.show()

    print('[', row['id'], '] Predicted: ', top_labels, ', Confident: ', top_confidents, ", truth: ", truthLabel)
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

print('acc: ', acc, ', top3: ', top3, ', top5: ', top5, ' / total: ', total)