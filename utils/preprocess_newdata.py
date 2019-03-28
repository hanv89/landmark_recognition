from PIL import Image
import argparse
import glob
import os
import random
import time
import numpy as np

parser = argparse.ArgumentParser(description='Landmark Detection Training then Finetune')

#Directories
parser.add_argument('--input', default='./data', type = str, help = 'input dir')
parser.add_argument('--output', type = str, help = 'output dir')
parser.add_argument('--type', default='jpg', type = str, help = 'file type')

args = parser.parse_args()
if args.output:
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    uris = glob.glob(args.input + '/*.' + args.type)
    imgs = []
    nuris = []
    for uri in uris:
        img = Image.open(uri)
        width, height = img.size
        dim = min(height, width)

        left = int(0)
        right = int(dim)
        upper = int(0)
        lower = int(dim)
        box = (left, upper, right, lower)
        print(img.size, dim, box)
        cropped_image1 = img.crop(box)
        cropped_image1.thumbnail((480, 480))
        imgs.append(cropped_image1)

        left = int((width - dim) / 2)
        right = int(left + dim)
        upper = int((height - dim) / 2)
        lower = int(upper + dim)
        box = (left, upper, right, lower)
        print(img.size, dim, box)
        cropped_image2 = img.crop(box)
        cropped_image2.thumbnail((480, 480))
        imgs.append(cropped_image2)

        left = int(width - dim)
        right = int(width)
        upper = int(height - dim)
        lower = int(height)
        box = (left, upper, right, lower)
        print(img.size, dim, box)
        cropped_image3 = img.crop(box)
        cropped_image3.thumbnail((480, 480))
        imgs.append(cropped_image3)

        filepath, f = os.path.split(uri)
        fn, ext = os.path.splitext(f)
        nfn = str(time.time())
        nuri1 = args.output + "/" + nfn + "_1" + args.rn + ext
        nuris.append(nuri1)
        nuri2 = args.output + "/" + nfn + "_2" + args.rn + ext
        nuris.append(nuri2)
        nuri3 = args.output + "/" + nfn + "_3" + args.rn + ext
        nuris.append(nuri3)

    for i in range(0,len(nuris)):
        imgs[i].save(nuris[i])
        print(nuris[i])


