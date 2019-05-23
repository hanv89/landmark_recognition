echo 'Validating '$1
python3 validate/create_submission.py --valdir=data/Public --output=submit_public.csv --savedmodel=$1 --class_index=labels/label.index.csv --size=$2
# python3 validate/create_submission.py --valdir=data/finalPrivateTest/private_test_3_9/ --output=submit_private.csv --savedmodel=$1 --class_index=labels/label.index.csv --size=$2