import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import utils

# tf.enable_eager_execution()

json_file = './data/train_val2018.json'
data_dir = './data/TrainVal/'

print('Loading data')
# A vector of filenames.
filenames, labels, count, eval_filenames, eval_labels, eval_count = utils.read_zalo(data_dir, json_file, 25)

print('Creating dataset')
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(utils._parse_function)
# dataset = dataset.repeat(5)
# dataset = dataset.batch(4)

print(dataset.output_types)
print(dataset.output_shapes)

# plt.figure(figsize=(10,10))
# i=0
# for images, labels in dataset.take(25):
#     i=i+1
#     plt.subplot(5,5,i)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(images, cmap=plt.cm.binary)
#     plt.xlabel(labels)
# plt.show()
# for images, labels in dataset.take(1):
#     print(images.shape)
#     # print(images*255)
#     plt.figure()
#     plt.imshow(images)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()
# iterator = dataset.make_one_shot_iterator()
# value = iterator.get_next()
# print(value[0].shape)
# print(value[0]/255)
# plt.figure()
# plt.imshow(value[0]/255)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# value = dataset[0][5]
# print(value[0].shape)
# print(value[0]/255)
# plt.figure()
# plt.imshow(value[0]/255)
# plt.colorbar()
# plt.grid(False)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=5, steps_per_epoch=30)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)