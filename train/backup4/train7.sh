python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=155 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_121 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=427 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=595 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=inception_v3 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=311 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=xception --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=132 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=resnet_50 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=175 --dropout1=0 --l21=0 \
                    --batch=32