# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# tf.enable_eager_execution()

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

# train_images = train_images / 255.0

# test_images = test_images / 255.0

plt.figure(figsize=(20,10))
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# plt.show()

def _parse_function(image, label):
    image_decoded = tf.reshape(image, [28,28,1])
    print(image_decoded.shape)
    image_resized = tf.image.resize_images(image_decoded, [24, 24]) / 255.0
    return image_resized, label

def _parse_function2(image, label):
    label_decoded = tf.convert_to_tensor(label, dtype=tf.int64)
    image_decoded = tf.reshape(image, [10000, 28,28,1])
    print(image_decoded.shape)
    image_resized = tf.image.resize_images(image_decoded, [24, 24]) / 255.0
    return image_resized, label_decoded

train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.map(_parse_function)
dataset = dataset.batch(1024).repeat()

test_images2, test_labels2 = _parse_function2(test_images, test_labels)

model = keras.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Flatten(input_shape=(7, 7, 16)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=50, steps_per_epoch=30)

test_loss, test_acc = model.evaluate(test_images2, test_labels2, steps=3)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images2, steps=3)

print(predictions[0])

print(np.argmax(predictions[0]))

print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()

# # Grab an image from the test dataset
# img2 = test_images2[0]
# img = test_images[0]

# print(img.shape)

# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img,0))

# print(img.shape)

# predictions_single = model.predict(img)

# print(predictions_single)

# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)

# print(np.argmax(predictions_single[0]))