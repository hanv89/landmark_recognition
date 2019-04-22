echo 'Validating '$1
python3 validate/validate.py --valdir=data/Public --input=data/publicTest.csv --savedmodel=$1 --class_index=labels/label.index.csv --size=$2
python3 validate/validate.py --valdir=data/finalPrivateTest/private_test_3_9/ --input=data/finalPrivateTest.csv --savedmodel=$1 --class_index=labels/label.index.csv --size=$2