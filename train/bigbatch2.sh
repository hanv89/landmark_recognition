python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=300 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=600 \
                    --finetune_lr2=0.0001 --finetune_epochs2=100 --finetune_steps_per_epoch2=600 \
                    --freeze=55 --dropout=0.5 --l2=0.5 \
                    --batch=128