import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import random
import json
import argparse
import time
import math
import sys
import os

# tf.enable_eager_execution() #only for test run

parser = argparse.ArgumentParser(description='Landmark Detection Training then Finetune')

#Directories
parser.add_argument('--data', default='./data', type = str, help = 'Data dir')
#Augmentation parameters
parser.add_argument('--horizontal_flip', type=bool, default=True)
parser.add_argument('--zoom_in', type=float, default=0)
parser.add_argument('--zoom_out', type=float, default=0)
parser.add_argument('--shear', type=float, default=0)
parser.add_argument('--width', type=float, default=0)
parser.add_argument('--height', type=float, default=0)
parser.add_argument('--rotate', type=int, default=0)
parser.add_argument('--channel', type=float, default=0)

args = parser.parse_args()


train_datagen = image.ImageDataGenerator(
  rescale=1./255,
  shear_range=args.shear,
  zoom_range=[1-args.zoom_in, 1+args.zoom_out],
  width_shift_range=args.width,
  height_shift_range=args.height,
  rotation_range=args.rotate,
  horizontal_flip=args.horizontal_flip,
  channel_shift_range=args.channel,
  validation_split=0.1,
  fill_mode='reflect')

train_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(299, 299),
  color_mode='rgb',
  batch_size=32,
  class_mode='sparse',
  shuffle=True,
  seed=1,
  subset='training'
)
validation_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(299, 299),
  color_mode='rgb',
  batch_size=32,
  class_mode='sparse',
  shuffle=True,
  seed=1,
  subset='validation'
)
# while True:
images, labels = next(train_generator)
plt.figure(figsize=(100,200))
for i in range (0,32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel(labels[i])
plt.show()


images, labels = next(validation_generator)
plt.figure(figsize=(100,200))
for i in range (0,32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel(labels[i])
plt.show()