import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet169,DenseNet201,DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetLarge,NASNetMobile
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import utils.utils as utils
import random
import json
import argparse
import time
import sys
import os

timestr = time.strftime('%Y%m%d-%H%M%S')
print('Training stated at ' ,timestr)

# tf.enable_eager_execution() #only for test run

parser = argparse.ArgumentParser(description='Landmark Detection Training then Finetune')

#Directories
parser.add_argument('--data', default='./data', type = str, help = 'Data dir')
parser.add_argument('--output', default='./output', type = str, help = 'Output dir')
parser.add_argument('--load_model', type = str, help = 'Saved model in h5 format, eg: ./models/model.h5')
parser.add_argument('--pretrained_model', type = str, help = 'Pretrained model, eg: ./pretrained/model.h5')

#Network configs
parser.add_argument('--net', default='inception_v3', choices=['resnet_50', 'inception_v3', 'xception', 'densenet_121', 'densenet_169', 'densenet_201', 'mobilenet_v2', 'nasnetmobile', 'nasnetlarge'], type = str, help = 'Network structure')
parser.add_argument('--freeze', default=-3, type = int, help = 'Number of layer to freeze in finetune')
parser.add_argument('--mode', default='train_then_finetune', choices=['print', 'train', 'finetune', 'train_then_finetune'], type = str, help = 'Train mode')

#Train parameters
parser.add_argument('--batch', default=32, type = int, help = 'batch size')
parser.add_argument('--train_epochs', default=2, type = int, help = 'number of train epoch')
parser.add_argument('--train_steps_per_epoch', default=5, type = int, help = 'number of step per train epoch')
parser.add_argument('--finetune_epochs', default=2, type = int, help = 'number of finetune epoch')
parser.add_argument('--finetune_steps_per_epoch', default=5, type = int, help = 'number of step per finetune epoch')

#Augmentation parameters
parser.add_argument('--validation_split', default=0.1, type=float, help='percent of training samples per class')
parser.add_argument('--horizontal_flip', type=bool, default=True)
parser.add_argument('--zoom', type=float, default=0.2)
parser.add_argument('--shear', type=float, default=0.2)
parser.add_argument('--width', type=float, default=0.2)
parser.add_argument('--height', type=float, default=0.2)
parser.add_argument('--rotate', type=int, default=20)
args = parser.parse_args()

output_dir = args.output + '/' + args.net + '-' + timestr
output_label = output_dir + '/label.index'
os.mkdir(output_dir)

train_output_dir = output_dir + '/train'
train_output_model = train_output_dir + '/model.h5'
train_output_log = train_output_dir + '/log'
os.mkdir(train_output_dir)

finetune_output_dir = output_dir + '/finetune'
finetune_output_model = finetune_output_dir + '/model.h5'
finetune_check_point_model = finetune_output_dir + '/check_point.h5'
finetune_output_log = finetune_output_dir + '/log'
os.mkdir(finetune_output_dir)

train_savedmodel_output_dir = train_output_dir + '/savedmodel'
finetune_savedmodel_output_dir = finetune_output_dir + '/savedmodel'
os.mkdir(train_savedmodel_output_dir)
os.mkdir(finetune_savedmodel_output_dir)

#specify input dimension
print('Network type: ', args.net)
if args.net.startswith('inception'):
  dim = 299
elif args.net.startswith('xception'):
  dim = 299
elif args.net.startswith('resnet'):
  dim = 224
elif args.net.startswith('densenet'):
  dim = 224
elif args.net.startswith('mobilenet'):
  dim = 224
elif args.net.startswith('nasnetlarge'):
  dim = 331
elif args.net.startswith('nasnetmobile'):
  dim = 224
else:
  print('Not supported network type')
  sys.exit()

#load data
train_datagen = image.ImageDataGenerator(
  rescale=1./255,
  shear_range=args.shear,
  zoom_range=args.zoom,
  width_shift_range=args.width,
  height_shift_range=args.height,
  rotation_range=args.rotate,
  horizontal_flip=args.horizontal_flip,
  validation_split=args.validation_split)

train_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(dim, dim),
  color_mode='rgb',
  batch_size=args.batch,
  class_mode='sparse',
  shuffle=True,
  seed=42,
  subset='training'
)

validation_generator = train_datagen.flow_from_directory(
  directory=args.data,
  target_size=(dim, dim),
  color_mode='rgb',
  batch_size=args.batch,
  class_mode='sparse',
  shuffle=True,
  seed=42,
  subset='validation'
)


with open(output_label + ".csv", 'w') as outfile:  
  outfile.write('\n'.join(train_generator.class_indices))

class_count = len(train_generator.class_indices)
class_index = {v: k for k, v in train_generator.class_indices.items()}

#create model
if not args.load_model and not args.mode == 'finetune':
  # create the base pre-trained model
  if args.pretrained_model: 
    base_model = keras.models.load_model(args.pretrained_model)
  elif args.net == 'inception_v3':
    base_model = InceptionV3(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'xception':
    base_model = Xception(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'resnet_50':
    base_model = ResNet50(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'densenet_121':
    base_model = DenseNet121(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'densenet_169':
    base_model = DenseNet169(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'densenet_201':
    base_model = DenseNet201(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'mobilenet_v2':
    base_model = MobileNetV2(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'nasnetlarge':
    base_model = NASNetLarge(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  elif args.net == 'nasnetmobile':
    base_model = NASNetMobile(input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
  else:
    print('Not supported network type')
    sys.exit()

  # add a global spatial average pooling layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
  # and a logistic layer
  predictions = Dense(class_count, activation='softmax')(x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # first: train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional InceptionV3 layers
  for layer in base_model.layers:
    layer.trainable = False

elif args.load_model and ( args.mode == 'print' or  args.mode == 'finetune' ):
  model = keras.models.load_model(args.load_model)
else:
  print('Not supported mode ', args.mode, ' when args.load_model is ', args.load_model)
  sys.exit()

if args.mode == 'print':
  print('Mode ',args.mode,': Printing network layers')
  for i, layer in enumerate(model.layers):
    print(i, layer.name)  
else:
  start = time.time()
  #output class indices to file
  with open(output_label, 'w') as outfile:  
    json.dump(class_index, outfile)
    
  if args.mode.startswith('train'):    
    print('Mode ',args.mode,': Training...')
    #compile training model
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

    callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=args.train_epochs/4, monitor='val_loss'),
      # Write TensorBoard logs
      tf.keras.callbacks.TensorBoard(log_dir=train_output_log)
    ]

    #train
    history = model.fit(train_generator, epochs=args.train_epochs, steps_per_epoch=args.train_steps_per_epoch, 
      validation_data=validation_generator, validation_steps=args.train_steps_per_epoch/10, 
      callbacks=callbacks)

    #save and print results
    model.save(train_output_model)
    tf.contrib.saved_model.save_keras_model(model, train_savedmodel_output_dir)    

    utils.print_history(history)

  if args.mode.endswith('finetune'):    
    print('Mode ',args.mode,': Finetune...')

    layers_count = len(model.layers)
    freeze_layers_count = (layers_count + args.freeze) % layers_count
    print('Freeze ',freeze_layers_count, ' layers')

    for layer in model.layers[:freeze_layers_count]:
      layer.trainable = False
    for layer in model.layers[freeze_layers_count:]:
      layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

    callbacks = [
      tf.keras.callbacks.ModelCheckpoint(finetune_check_point_model,monitor='val_loss',save_best_only=True),
      tf.keras.callbacks.EarlyStopping(patience=args.finetune_epochs/4, monitor='val_loss'),
      # Write TensorBoard logs
      tf.keras.callbacks.TensorBoard(log_dir=finetune_output_log)
    ]

    #train
    history = model.fit(train_generator, epochs=args.finetune_epochs, steps_per_epoch=args.finetune_steps_per_epoch, 
      validation_data=validation_generator, validation_steps=args.finetune_steps_per_epoch/10, 
      callbacks=callbacks)

    #save and print results
    model.save(finetune_output_model)
    tf.contrib.saved_model.save_keras_model(model, finetune_savedmodel_output_dir)  

    utils.print_history(history)

  exec_time = time.time() - start
  print("[", timestr, "] exec time: ", exec_time)
