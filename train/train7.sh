python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=4 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=55 --dense1=1280 \
                    --dropout0=0.5 --validation_split=0.1 --seed=1 --use_class_weight=True
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=4 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=55 --dense1=1280 \
                    --dropout0=0.5 --validation_split=0.1 --seed=2 --use_class_weight=True
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=4 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=55 --dense1=1280 \
                    --dropout0=0.5 --validation_split=0.1 --seed=3 --use_class_weight=True
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=4 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=55 --dense1=1280 \
                    --dropout0=0.5 --validation_split=0.1 --seed=4 --use_class_weight=True
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=4 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=55 --dense1=1280 \
                    --dropout0=0.5 --validation_split=0.1 --seed=5 --use_class_weight=True