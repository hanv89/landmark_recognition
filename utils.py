import tensorflow as tf
import json
import os
import random
import imghdr

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
        if os.path.getsize(filename) == 0: # zero-byte files 
            continue 
        if imghdr.what(filename) not in ['jpeg', 'png', 'gif']: # invalid image files
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

        
    return filenames, labels, count, val_filenames, val_labels, val_count


# From python basic tutorials
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [240, 240])
    return image_resized, label
