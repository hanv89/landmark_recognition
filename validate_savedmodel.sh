echo 'Validating '$1
python3 validate.py --valdir=data/Public --input=data/publicTest.csv --savedmodel=$1/finetune/savedmodel/$2 --class_index=output/label.index.csv --size=$3
python3 validate.py --valdir=data/finalPrivateTest/private_test_3_9/ --input=data/finalPrivateTest.csv --savedmodel=$1/finetune/savedmodel/$2 --class_index=output/label.index.csv --size=$3