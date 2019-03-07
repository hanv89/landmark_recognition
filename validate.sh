echo 'Validating '$1
python3 validate.py --valdir=data/Public --input=data/publicTest.csv --model=$1/finetune/check_point.h5 --class_index=$1/label.index.csv --size=$2
python3 validate.py --valdir=data/finalPrivateTest/private_test_3_9/ --input=data/finalPrivateTest.csv --model=$1/finetune/check_point.h5 --class_index=$1/label.index.csv --size=$2