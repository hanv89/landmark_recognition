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
filenames, labels, count, val_filenames, val_labels, val_count = utils.read_zalo(data_dir, json_file)

print('Creating dataset', count)
labels = tf.convert_to_tensor(labels, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(utils._parse_function28)
dataset = dataset.batch(128).repeat()

print(dataset.output_types)
print(dataset.output_shapes)

print('Creating val dataset', val_count)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int64)
val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val_dataset = val_dataset.map(utils._parse_function28)
val_dataset = val_dataset.batch(128).repeat()

print(val_dataset.output_types)
print(val_dataset.output_shapes)

# plt.figure(figsize=(10,20))
# for images, labels in dataset:
#     for i in range(800):
#         plt.subplot(20,40,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images[i], cmap=plt.cm.binary)
#         plt.xlabel(labels[i])
#     plt.show()
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
    # keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(1024, activation=tf.nn.relu)

    keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Flatten(input_shape=(7, 7, 64)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(103, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy' 
                        # ,keras.metrics.sparse_top_k_categorical_accuracy
                ])

model.fit(dataset, epochs=10, steps_per_epoch=30,
            validation_data=val_dataset,
            validation_steps=3)


eval_loss, eval_acc = model.evaluate(val_dataset, steps=3)

print('Test accuracy:', eval_acc)