python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=resnet \
                    --train_epochs=2 --train_steps_per_epoch=120 \
                    --finetune_epochs=12 --finetune_steps_per_epoch=120 \
                    --freeze=1 --dropout=0.4 \
                    --workers=2 --batch=4