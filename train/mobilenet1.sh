python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=160 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=320 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=320 \
                    --freeze=55 --dropout=0.5 --l2=0 \
                    --batch=256
python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=160 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=320 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=320 \
                    --freeze=91 --dropout=0.5 --l2=0 \
                    --batch=256