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
open (args.output,'wb').write(tfmodel)

# tflite_convert --output_file=./output/densenet_169-20190223-105925/quantized_densenet_169-20190223-105925.tflite --keras_model_file=./output/densenet_169-20190223-105925/finetune/model.h5 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=127 --default_ranges_min=0 --default_ranges_max=6