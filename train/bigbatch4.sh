python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=300 \
                    --finetune_lr1=0.0004 --finetune_epochs1=10 --finetune_steps_per_epoch1=600 \
                    --finetune_lr2=0.0002 --finetune_epochs2=60 --finetune_steps_per_epoch2=600 \
                    --freeze=367 --dropout=0.5 --l2=0 \
                    --batch=128

python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet --workers=2 \
                    --train_epochs=4 --train_steps_per_epoch=300 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=600 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=600 \
                    --freeze=51 --dropout=0.5 --l2=0.1 \
                    --batch=128