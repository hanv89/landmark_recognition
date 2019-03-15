python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=5 --train_steps_per_epoch=300 \
                    --finetune_lr1=0.0002 --finetune_epochs1=20 --finetune_steps_per_epoch1=300 \
                    --finetune_lr2=0.0004 --finetune_epochs2=80 --finetune_steps_per_epoch2=300 \
                    --freeze=91 --dropout=0.5 --l2=0 \
                    --batch=256 --crop=0 --width=0.2 --height=0.2