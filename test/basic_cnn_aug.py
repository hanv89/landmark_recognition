import tensorflow as tf
from tensorflow import keras
import random

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import utils

# tf.enable_eager_execution()

data_dir = r'./data/TrainVal/'

train_datagen = image.ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
  directory=data_dir,
  target_size=(128, 128),
  color_mode="rgb",
  batch_size=32,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="training"
)

validation_generator = train_datagen.flow_from_directory(
  directory=data_dir,
  target_size=(128, 128),
  color_mode="rgb",
  batch_size=32,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="validation"
)

# this is the model we will train
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(128, 128, 3),filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Flatten(input_shape=(16, 16, 128)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(103, activation=tf.nn.softmax)
])

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./my_basic_cnn/20190119/logs')
]
# train the model on the new data for a few epochs
history = model.fit(train_generator, epochs=200, steps_per_epoch=2000, 
  validation_data=validation_generator, validation_steps=100, 
  callbacks=callbacks)

print('max_val_acc: ',max(history.history['val_acc']))
print('min_val_acc: ',min(history.history['val_acc']))
print('average_val_acc: ',utils.average(history.history['val_acc']))
print('max_val_loss: ',max(history.history['val_loss']))
print('min_val_loss: ',min(history.history['val_loss']))
print('average_val_loss: ',utils.average(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

model.save('my_basic_cnn/20190119/basic_cnn-model-20190119.h5')