#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv, io
from urllib.request import urlopen
from PIL import Image

def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header

def ParseDataWLabel(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:3] for line in csvreader]
  return key_url_list[1:]  # Chop off header

def DownloadImage(key_url):
  out_dir = sys.argv[2]
  (key, url, label) = key_url
  cate_dir = os.path.join(out_dir, '%s' % label)
  filename = os.path.join(cate_dir, '%s.jpg' % key)

  if not os.path.exists(cate_dir):
    os.makedirs(cate_dir)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    response = urlopen(url)
    image_data = response.read()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(io.BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.thumbnail((600, 600))
  except:
    print('Warning: Failed to resize image %s' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return


def Run():
  if len(sys.argv) != 3:
    print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseDataWLabel(data_file)
  pool = multiprocessing.Pool(processes=32)
  pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()