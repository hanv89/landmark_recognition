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
parser.add_argument('--dev_output', default='./dev.csv', type = str, help = 'Validate csv file')
parser.add_argument('--train_output', default='./train.csv', type = str, help = 'Train csv file')
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
# images, labels = next(train_generator)
# plt.figure(figsize=(100,200))
# for i in range (0,32):
#     plt.subplot(4,8,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(images[i])
#     plt.xlabel(labels[i])
# plt.show()

def extract(str):
  str = str.replace('.jpg','')
  str = str.split('/')
  return str[1] +','+ str[0]

with open(args.dev_output, 'w') as outfile:  
  outfile.write('\n'.join(list(map(extract, validation_generator.filenames))))
with open(args.train_output, 'w') as outfile:  
  outfile.write('\n'.join(list(map(extract, train_generator.filenames))))
    
# for i in validation_generator:
#     idx = (validation_generator.batch_index - 1) * validation_generator.batch_size
#     print(validation_generator.filenames[idx : idx + validation_generator.batch_size],",",validation_generator.labels[idx : idx + validation_generator.batch_size])

# while True:
#   images, labels = next(validation_generator)
# plt.figure(figsize=(100,200))
# for i in range (0,32):
#     plt.subplot(4,8,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(images[i])
#     plt.xlabel(labels[i])
# plt.show()