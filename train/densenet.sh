python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=139 --dropout=0.5 \
                    --workers=2 --batch=64 --crop=0.2 --width=0 --height=0