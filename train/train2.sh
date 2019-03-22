python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --load_model=./output/nasnetmobile-20190322-134348/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=10 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0002 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 \
                    --batch=64