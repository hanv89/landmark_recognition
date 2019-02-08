import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument('--class_index', default='index_to_class.json', type = str, help = 'index map')
parser.add_argument('--input', default='image.jpg', type = str, help = 'input image')
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

img = image.load_img(args.input, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)

top = preds[0].argsort()[-3:][::-1]
top = list(map(lambda x: index_to_class[str(x)], top))

plt.figure(figsize=(10,10))
# plt.xticks([])
# plt.yticks([])
plt.imshow(x[0])
# plt.imshow(img, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.xlabel(top)
plt.show()

print('Predicted:', top)
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]