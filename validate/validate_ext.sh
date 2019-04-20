echo 'Validating '$1
python3 validate/validate.py --valdir=data/uni/zuni --input=data/uni/zuni.csv --model=$1 --class_index=labels/label.index.ext.csv --size=$2
python3 validate/validate.py --valdir=data/zbk/zbk_private --input=data/zbk/zbk.csv --model=$1 --class_index=labels/label.index.ext.csv --size=$2
python3 validate/validate.py --valdir=data/Public --input=data/publicTest.csv --model=$1 --class_index=labels/label.index.ext.csv --size=$2
python3 validate/validate.py --valdir=data/finalPrivateTest/private_test_3_9/ --input=data/finalPrivateTest.csv --model=$1 --class_index=labels/label.index.ext.csv --size=$2