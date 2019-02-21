import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import utils
import random
import json
import argparse
import time
import sys

timestr = time.strftime("%Y%m%d-%H%M%S")
print("Training stated at " ,timestr)

# tf.enable_eager_execution() #only for test run

parser = argparse.ArgumentParser(description='Landmark Detection Training')

#Directories
parser.add_argument('--data', default="./data", type = str, help = 'Data dir')
parser.add_argument('--output', default="./output", type = str, help = 'Output dir')
parser.add_argument('--load_model', default='./models/model.h5', type = str, help = 'Saved model in h5 format')

#Network configs
parser.add_argument('--net', default='inception_v3', type = str, help = 'Network structure: inception_v3, resnet_v2_50, resnet_v2_152')
parser.add_argument('--freeze', default=-3, type = int, help = 'Number of layer to freeze')
parser.add_argument('--mode', default=0, type = int, help = 'Train mode: 0 for print network structure; 1 for fast train; 2 finetune only, 3 for both')

#Train parameters
parser.add_argument('--optimizer', default='adam', type = str, help = 'Optimizer')
parser.add_argument('--lr', default=1e-2, type=float, help = 'learning rate')
parser.add_argument('--batch', default=32, type = int, help = 'batch size')
parser.add_argument('--epochs', default=5, type = int, help = 'number of epoch')
parser.add_argument('--steps_per_epoch', default=10, type = int, help = 'number of step per epoch')

#Augmentation parameters
parser.add_argument('--validation_split', default=0.1, type=float, help="percent of training samples per class")
parser.add_argument('--horizontal_flip', type=bool, default=True)
parser.add_argument('--zoom', type=float, default=0.2)
parser.add_argument('--shear', type=float, default=0.2)
args = parser.parse_args()

output_dir = args.output + '/' + args.net + '-' + timestr
output_label = output_dir + '/label.index'
output_model = output_dir + '/model.h5'
output_log = output_dir + '/log'

if args.net.startswith('inception'):
  dim = 299
elif args.net.startswith('resnet'):
  dim = 224
else:
  print("Not supported network type")
  sys.exit()

train_datagen = image.ImageDataGenerator(
  rescale=1./255,
  shear_range=args.shear,
  zoom_range=args.zoom,
  horizontal_flip=args.horizontal_flip,
  validation_split=args.validation_split)

train_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(dim, dim),
  color_mode="rgb",
  batch_size=args.batch,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="training"
)

validation_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(dim, dim),
  color_mode="rgb",
  batch_size=args.batch,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="validation"
)

class_count = len(train_generator.class_indices)
class_index = {v: k for k, v in train_generator.class_indices.items()}
with open(output_label), 'w') as outfile:  
  json.dump(class_index, outfile)

if not args.load_model:
  # create the base pre-trained model
  if args.net == "inception_v3":
    base_model = InceptionV3(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  else:
    print("Not supported network type")
    sys.exit()

  # add a global spatial average pooling layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer
  predictions = Dense(class_count, activation='softmax')(x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)
else:
  model = keras.models.load_model(args.load_model)

if args.mode == 0:
  for i, layer in enumerate(model.layers):
   print(i, layer.name)
else:
  if args.optimizer == "momentum":
    optimizer = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

  layers_count = len(model.layers)
  freeze_layers_count = (layers_count + args.freeze) % layers_count

  for layer in model.layers[:freeze_layers_count]:
    layer.trainable = False
  for layer in model.layers[freeze_layers_count:]:
    layer.trainable = True

  # compile the model (should be done *after* setting layers to non-trainable)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

  callbacks = [
    # Interrupt training if `val_loss` stops improving for over few epochs
    tf.keras.callbacks.EarlyStopping(patience=epochs/10, monitor='val_loss'),
    # Write TensorBoard logs
    tf.keras.callbacks.TensorBoard(log_dir=output_log)
  ]

  history = model.fit(train_generator, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, 
    validation_data=validation_generator, validation_steps=args.steps_per_epoch/10, 
    callbacks=callbacks)

  model.save(output_model)

  print('max_val_acc: ',max(history.history['val_acc']))
  print('min_val_acc: ',min(history.history['val_acc']))
  print('average_val_acc: ',utils.average(history.history['val_acc']))
  print('max_val_top_3: ',max(history.history['val_top_3_accuracy']))
  print('min_val_top_3: ',min(history.history['val_top_3_accuracy']))
  print('average_val_top_3: ',utils.average(history.history['val_top_3_accuracy']))
  print('max_val_loss: ',max(history.history['val_loss']))
  print('min_val_loss: ',min(history.history['val_loss']))
  print('average_val_loss: ',utils.average(history.history['val_loss']))
  print('train_acc: ',max(history.history['acc']))
  print('train_loss: ',min(history.history['loss']))
  print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))