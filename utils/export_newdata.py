import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Landmark Detection Training then Finetune')

#Directories
parser.add_argument('--input', default='./data', type = str, help = 'input dir')
parser.add_argument('--output', type = str, help = 'output dir')
parser.add_argument('--type', default='jpg', type = str, help = 'file type')
parser.add_argument('--rn', default='a', type = str, help = 'add to file name')

args = parser.parse_args()
if args.output:
    uris = glob.glob(args.input + '/*.' + args.type)
    fns = []
    for uri in uris:
        filepath, f = os.path.split(uri)
        fn, ext = os.path.splitext(f)
        fns.append(fn + ',993')
    with open(args.output, 'w') as outfile:  
        outfile.write('\n'.join(fns))