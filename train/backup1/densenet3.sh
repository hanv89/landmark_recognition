# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
#                     --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
#                     --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=1280 \
#                     --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=1280 \
#                     --freeze=139 --dropout=0.5 --l2=0.1 \
#                     --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
#                     --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=640 \
#                     --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=1280 \
#                     --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=1280 \
#                     --freeze=139 --dropout=0.5 --l2=0.01 \
#                     --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=160 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=320 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=320 \
                    --freeze=532 --dropout=0 --l2=0 \
                    --batch=256
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=320 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=640 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=640 \
                    --freeze=532 --dropout=0 --l2=0 \
                    --batch=128
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --workers=2 \
                    --train_lr=0.0004 --train_epochs=4 --train_steps_per_epoch=320 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=640 \
                    --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=640 \
                    --freeze=532 --dropout=0.5 --l2=0 \
                    --batch=128