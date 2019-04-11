
from clint.textui import progress
import argparse
import requests
import zipfile
import imghdr
import glob
import time
import math
import sys
import os

parser = argparse.ArgumentParser(description='Landmark Recognition Get Data')
parser.add_argument('--data', default='./data', type = str, help = 'Data dir')
parser.add_argument('--data_src', type = str, help = 'Data source')
args = parser.parse_args()

if not args.data_src:
    print('Data source is not specified!')
else:
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    if os.path.exists(args.data) and os.path.isdir(args.data):
        if not os.listdir(args.data):
            print("Directory is empty, getting data...")
            
            url = args.data_src              
            filepath, fn = os.path.split(url)  
            local_fn = args.data + "/" + fn      
            r = requests.get(url, stream=True)
            
            with open(local_fn, "wb") as local_file:            
                total_length = int(r.headers.get('content-length'))            
                for ch in progress.bar(r.iter_content(chunk_size = 8192), expected_size=(total_length/8192) + 1):            
                    if ch:            
                        local_file.write(ch)
            
            print("Extracting...")
            zip_file = zipfile.ZipFile(local_fn, 'r')
            zip_file.extractall(args.data)
            zip_file.close()
            os.remove(local_fn)

            print("Removing error files")
            for filename in glob.iglob(args.data+'/**/*.jpg', recursive=True):
                if os.path.getsize(filename) == 0: # zero-byte files 
                    os.remove(filename)
                    print("Zero: ", filename)
                    continue 
                if imghdr.what(filename) not in ['jpeg', 'png', 'gif']: # invalid image files
                    os.remove(filename)
                    print("Invalid: ", filename)
                    continue

            print("Done")
        else:    
            print("Directory already has data")
    else:
        print("Cannot access dir", args.data)     