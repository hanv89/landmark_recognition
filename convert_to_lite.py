from tensorflow.contrib import lite
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='model.h5', type = str, help = 'input h5 model')
parser.add_argument('--output', default='model.tflite', type = str, help = 'output tflite model')
args = parser.parse_args()

print(args.input)
print(args.output)

converter = lite.TFLiteConverter.from_keras_model_file(args.input)
tfmodel = converter.convert()
open (args.output).write(tfmodel)