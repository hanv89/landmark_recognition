python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dense1=1024 \
                    --dropout0=0.5 --use_class_weight=True
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=56 --dense1=1024 \
                    --dropout0=0.5
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=139 --dense1=1024 \
                    --dropout0=0.5
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dense1=512 \
                    --dropout0=0.5
python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dense1=256 \
                    --dropout0=0.5