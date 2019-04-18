from tensorflow.contrib import lite
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='model.tflite', type = str, help = 'input tflite model')
parser.add_argument('--output', default='quantized_model.tflite', type = str, help = 'output quantized model')
args = parser.parse_args()

print(args.input)
print(args.output)

converter = lite.TocoConverter.from_saved_model(args.input)
converter.post_training_quantize = True
tflite_quantized_model = converter.convert()
open(args.output, "wb").write(tflite_quantized_model)
