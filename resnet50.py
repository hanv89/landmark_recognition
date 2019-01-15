import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import utils
import random

# tf.enable_eager_execution()

json_file = './data/train_val2018.json'
data_dir = './data/TrainVal/'

print('Loading data')
# A vector of filenames.
filenames, labels, count, val_filenames, val_labels, val_count = utils.read_zalo(data_dir, json_file)

# plt.figure(figsize=(20,10))
# for i in range(25):
#     image, label = utils._parse_function(filenames[i], labels[i])
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image.numpy(), cmap=plt.cm.binary)
#     plt.xlabel(filenames[i] + ":" + str(label))
# plt.show()

print('Creating dataset', count)
labels = tf.convert_to_tensor(labels, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(utils._parse_function224)
dataset = dataset.batch(64).repeat()

print(dataset.output_types)
print(dataset.output_shapes)

print('Creating val dataset', val_count)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int64)
val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val_dataset = val_dataset.map(utils._parse_function224)
val_dataset = val_dataset.batch(64).repeat()

print(val_dataset.output_types)
print(val_dataset.output_shapes)

# this could also be the output a different Keras model or layer
# input_tensor = Input(shape=(240, 240, 3))  # this assumes K.image_data_format() == 'channels_last'

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

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

# train the model on the new data for a few epochs

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./my_resnet50/2019011/logs')
]

history = model.fit(dataset, epochs=2, steps_per_epoch=2, validation_data=val_dataset, validation_steps=3, callbacks=callbacks)

model.save('my_resnet50/20190115/resnet-model-20190115.h5')

print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:123]:
   layer.trainable = False
for layer in model.layers[123:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy', utils.top_3_accuracy])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./my_resnet50/2019011/logs')
]

history = model.fit(dataset, epochs=100, steps_per_epoch=1000, validation_data=val_dataset, validation_steps=3, callbacks=callbacks)

model.save('my_resnet50/20190115/resnet-remodel-20190115.h5')

print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))