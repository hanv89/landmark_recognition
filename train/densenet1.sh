python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=320 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=640 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=640 \
                    --freeze=139 --dropout=0.5 --l2=0.05 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=320 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=640 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=640 \
                    --freeze=51 --dropout=0.5 --l2=0.05 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=320 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=640 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=640 \
                    --freeze=367 --dropout=0.5 --l2=0.05 \
                    --batch=64
