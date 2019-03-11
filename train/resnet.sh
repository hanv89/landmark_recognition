python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=resnet_50 \
                    --train_epochs=2 --train_steps_per_epoch=12 \
                    --finetune_epochs=12 --finetune_steps_per_epoch=120 \
                    --freeze=1 --dropout=0.4 \
                    --workers=2 --batch=4

# python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=resnet_50 \
#                     --train_epochs=10 --train_steps_per_epoch=1200 \
#                     --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
#                     --freeze=0 --dropout=0.5 \
#                     --workers=2 --batch=64