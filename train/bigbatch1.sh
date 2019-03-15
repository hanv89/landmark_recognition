python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=5 --train_steps_per_epoch=150 \
                    --finetune_lr1=0.0002 --finetune_epochs1=40 --finetune_steps_per_epoch1=150 \
                    --finetune_lr2=0.0002 --finetune_epochs2=60 --finetune_steps_per_epoch2=150 \
                    --freeze=91 --dropout=0.5 --l2=0 \
                    --batch=512 --crop=0 --width=0.2 --height=0.2