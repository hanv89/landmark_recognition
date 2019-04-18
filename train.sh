#!/bin/bash
#Zalo ai challenge complete training run
#check data availablity and get if required
python3 utils/get_data.py --data=./data --data_src=https://dl.challenge.zalo.ai/landmark/train_val2018.zip
#train
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dropout0=0.5