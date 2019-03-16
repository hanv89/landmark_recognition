
from tensorflow.keras.preprocessing import image
import imgaug as ia
from imgaug import augmenters as iaa
import argparse
import glob
import os
import random
import numpy as np

ia.seed(random.randint(0, 100))

parser = argparse.ArgumentParser(description='Landmark Detection Training then Finetune')

#Directories
parser.add_argument('--input', default='./data', type = str, help = 'input dir')
parser.add_argument('--output', type = str, help = 'output dir')
parser.add_argument('--type', default='jpg', type = str, help = 'file type')
parser.add_argument('--rn', default='a', type = str, help = 'add to file name')

args = parser.parse_args()
if args.output:
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    seq = iaa.Sequential([
        iaa.Crop(percent=(0.1, 0.3)), # random crops    
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    uris = glob.glob(args.input + '/*.' + args.type)
    imgs = []
    nuris = []
    for uri in uris:
        img = image.load_img(uri, target_size=(480,480))
        img = image.img_to_array(img)
        imgs.append(img)

        filepath, f = os.path.split(uri)
        fn, ext = os.path.splitext(f)
        nuri = args.output + "/" + fn + args.rn + ext
        nuris.append(nuri)

    images_aug = seq.augment_images(imgs)

    for i in range(0,len(nuris)):
        image.array_to_img(images_aug[i]).save(nuris[i])
        print(nuris[i])