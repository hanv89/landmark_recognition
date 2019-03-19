# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
#                     --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
#                     --finetune_lr1=0.0004 --finetune_epochs1=10 --finetune_steps_per_epoch1=1280 \
#                     --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=1280 \
#                     --freeze=139 --dropout=0.5 --l2=0.05 \
#                     --batch=64 \
#                     --horizontal_flip=False --zoom_in=0 --zoom_out=0 --shear=0 --width=0 --height=0 --rotate=0 --channel=0
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=1280 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=1280 \
                    --freeze=55 --dropout=0 --l2=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=1280 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=1280 \
                    --freeze=91 --dropout=0 --l2=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=1280 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=1280 \
                    --freeze=117 --dropout=0 --l2=0 \
                    --batch=64