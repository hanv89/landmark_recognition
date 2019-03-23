python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0002 --finetune_epochs2=120 --finetune_steps_per_epoch2=400 \
                    --freeze=55 --dropout1=0 --l21=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=256
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=128
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=32
