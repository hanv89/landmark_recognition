python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=55 --dropout=0.5 \
                    --workers=2 --batch=64 --crop=0