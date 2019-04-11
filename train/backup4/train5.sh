python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.00005 --finetune_epochs1=80 --finetune_steps_per_epoch1=800 --finetune_lr_decay=0.1 \
                    --finetune_lr2=0.001 --finetune_epochs2=0 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dropout0=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=80 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dropout0=0 --l21=0 \
                    --batch=32 --use_class_weight=True
