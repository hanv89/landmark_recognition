import tensorflow as tf
import json
import os
import random
import imghdr
import numpy

def get_fns_lbs(base_dir, json_file, pickle_fn = 'mydata.p', force = False):    
    pickle_fn = base_dir + pickle_fn 
    # pdb.set_trace() 
    if os.path.isfile(pickle_fn) and not force:
        mydata = pickle.load(open(pickle_fn, 'rb'))
        fns = mydata['fns']
        lbs = mydata['lbs']
        cnt = mydata['cnt']
        return fns, lbs, cnt

    f = open(json_file, 'r')
    line = f.readlines()[0] # only one line 
    end = 0 
    id_marker = '\\"id\\": '
    cate_marker = '\\"category\\": '
    cnt = 0 
    fns = [] # list of all image filenames
    lbs = [] # list of all labels
    while True:
        start0 = line.find(id_marker, end)
        if start0 == -1: break 
        start_id = start0 + len(id_marker)
        end_id = line.find(',', start_id) 

        start0 = line.find(cate_marker, end_id)
        start_cate = start0 + len(cate_marker)
        end_cate = line.find('}', start_cate)

        end = end_cate
        cnt += 1
        cl = line[start_cate:end_cate]
        fn = base_dir + cl + '/' + line[start_id:end_id] + '.jpg'
        if os.path.getsize(fn) == 0: # zero-byte files 
            continue 
        lbs.append(int(cl))
        fns.append(fn)

    # pdb.set_trace()
    mydata = {'fns':fns, 'lbs':lbs, 'cnt':cnt}
    pickle.dump(mydata, open(pickle_fn, 'wb'))
    print(os.path.isfile(pickle_fn))

    return fns, lbs, cnt 

def read_zalo(base_dir, json_file, max = 0):
    input_file = open(json_file)
    json_array = json.load(input_file)
    
    count = 0 
    filenames = [] # list of all image filenames
    labels = [] # list of all labels

    val_count = 0 
    val_filenames = [] # list of all image filenames
    val_labels = [] # list of all labels

    for item in json_array:
        cate = item['category']
        name = item['id']
        filename = base_dir + str(cate) + '/' + str(name) + '.jpg'
        if os.path.exists(filename) == 0: # removed files 
            print("Removed: ", filename)
            continue 
        if os.path.getsize(filename) == 0: # zero-byte files 
            os.remove(filename)
            print("Zero: ", filename)
            continue 
        if imghdr.what(filename) not in ['jpeg', 'png', 'gif']: # invalid image files
            os.remove(filename)
            print("Invalid: ", filename)
            continue

        # print(cate, " > ", filename)

        if random.uniform(0.0, 1.0) > 0.1:
            filenames.append(filename)
            labels.append(cate)
            count+=1
        else:
            val_filenames.append(filename)
            val_labels.append(cate)
            val_count+=1
        if max > 0 and count >= max:
            break
    
    filenames, labels = unison_shuffled_copies(filenames, labels)

    return filenames, labels, count, val_filenames, val_labels, val_count


# From python basic tutorials
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label, size=240):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [size, size]) / 255.0
    return image_resized, label

def _parse_function240(filename, label):
    return _parse_function(filename, label, 240)

def _parse_function224(filename, label):
    return _parse_function(filename, label, 224)

def _parse_function299(filename, label):
    return _parse_function(filename, label, 299)
    
def _parse_function28(filename, label):
    return _parse_function(filename, label, 28)

def unison_shuffled_copies(a, b):
    a = numpy.array(a)
    b = numpy.array(b)
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3) 

def average(a): 
    return numpy.mean(a)

def print_history(history):
    print('val_acc: [',
        min(history.history['val_acc']), 
        ',', max(history.history['val_acc']), 
        '], average: ', average(history.history['val_acc']))

    # print('max_val_top_3: [',
    #     min(history.history['val_top_3_accuracy']), 
    #     ',', max(history.history['val_top_3_accuracy']), 
    #     '] average: ', average(history.history['val_top_3_accuracy']))
        
    print('max_val_loss: [', 
        min(history.history['val_loss']), 
        ',', max(history.history['val_loss']),
        '] average: ', average(history.history['val_loss']))

    print('train_acc: ',max(history.history['acc']))
    print('train_loss: ',min(history.history['loss']))
    print('train/val loss ratio: ', min(history.history['loss'])/min(history.history['val_loss']))

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = numpy.random.randint(0, width - dx + 1)
    y = numpy.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def random_crop_tf(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dim = math.min(height, width)
    # TensorFlow. 'x' = A placeholder for an image.
    desire_size = [random_crop_size, random_crop_size, 3]
    x = tf.placeholder(dtype = tf.float32, shape = desire_size)
    # Use the following commands to perform random crops
    new_dim = numpy.random.randint(random_crop_size, dim)

    crop_size = [new_dim, new_dim, 3]
    seed = np.random.randint(1234)
    x = tf.random_crop(x, size = crop_size, seed = seed)
    
    output = tf.images.resize_images(x, size = desire_size)
    return output


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = numpy.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

