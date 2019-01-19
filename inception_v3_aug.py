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
  target_size=(299, 299),
  color_mode="rgb",
  batch_size=64,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="training"
)

validation_generator = train_datagen.flow_from_directory(
  directory=data_dir,
  target_size=(299, 299),
  color_mode="rgb",
  batch_size=64,
  class_mode="sparse",
  shuffle=True,
  seed=42,
  subset="validation"
)

# this could also be the output a different Keras model or layer
# input_tensor = Input(shape=(240, 240, 3))  # this assumes K.image_data_format() == 'channels_last'

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(103, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])


callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./my_inception_v3/20190119/logs')
]
# train the model on the new data for a few epochs
history = model.fit(train_generator, epochs=20, steps_per_epoch=1000, 
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

model.save('my_inception_v3/20190119/inception_v3-model-20190119.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./my_inception_v3/20190119/relogs')
]
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit(train_generator, epochs=20, steps_per_epoch=1000, 
  validation_data=validation_generator, validation_steps=100, 
  callbacks=callbacks)

model.save('my_inception_v3/20190119/inception_v3-remodel-20190119.h5')

print('max_val_acc: ',max(history.history['val_acc']))
print('min_val_acc: ',min(history.history['val_acc']))
print('average_val_acc: ',utils.average(history.history['val_acc']))
print('max_val_loss: ',max(history.history['val_loss']))
print('min_val_loss: ',min(history.history['val_loss']))
print('average_val_loss: ',utils.average(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))