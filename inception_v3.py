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

tf.enable_eager_execution()

json_file = './data/train_val2018.json'
data_dir = './data/TrainVal/'

print('Loading data')
# A vector of filenames.
filenames, labels, count, val_filenames, val_labels, val_count = utils.read_zalo(data_dir, json_file)

# plt.figure(figsize=(20,10))
# for i in range(50):
#     j = random.randint(0,70000)
#     image, label = utils._parse_function(filenames[j], labels[j])
#     plt.subplot(5,10,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image.numpy(), cmap=plt.cm.binary)
#     plt.xlabel(label)
# plt.show()

print('Creating dataset', count)
# labels = tf.convert_to_tensor(labels, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(utils._parse_function)
dataset = dataset.batch(128).repeat()

print(dataset.output_types)
print(dataset.output_shapes)

print('Creating val dataset', val_count)
# val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int64)
val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val_dataset = val_dataset.map(utils._parse_function)
val_dataset = val_dataset.batch(128).repeat()

print(val_dataset.output_types)
print(val_dataset.output_shapes)

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(240, 240, 3))  # this assumes K.image_data_format() == 'channels_last'

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

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
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')

# train the model on the new data for a few epochs

model.fit(dataset, epochs=10, steps_per_epoch=500)

            # ,validation_data=val_dataset,
            # validation_steps=3)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# model.fit_generator(...)