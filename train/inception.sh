python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=inception_v3 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=249 --dropout=0 \
                    --workers=2 --batch=64 --crop=0