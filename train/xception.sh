python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=xception \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=56 --dropout=0 \
                    --workers=2 --batch=64 --crop=0