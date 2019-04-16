python3 finetune.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --load_model=./output/nasnetmobile-20190412-092025/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=0 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 \
                    --freeze=0 --dense1=1024 \
                    --dropout0=0.5