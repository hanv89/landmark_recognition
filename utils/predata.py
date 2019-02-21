import tensorflow as tf
from tensorflow import keras
import utils

print(tf.VERSION)
print(tf.keras.__version__)

json_file = './data/train_val2018.json'
data_dir = './data/TrainVal/'

print('Pre data')
# A vector of filenames.
filenames, labels, count, val_filenames, val_labels, val_count = utils.read_zalo(data_dir, json_file)
