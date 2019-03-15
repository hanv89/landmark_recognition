python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=5 --train_steps_per_epoch=600 \
                    --finetune_lr1=0.0005 --finetune_epochs1=40 --finetune_steps_per_epoch1=600 \
                    --finetune_lr2=0.0004 --finetune_epochs2=60 --finetune_steps_per_epoch2=600 \
                    --freeze=117 --dropout=0.5 --l2=0 \
                    --batch=128 --crop=0 --width=0.2 --height=0.2