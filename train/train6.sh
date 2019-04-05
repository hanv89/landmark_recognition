python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=55 --dropout0=0.6 --l21=0.01 \
                    --batch=32 --dense1=1280
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dropout0=0.6 --l21=0.01 \
                    --batch=32 --dense1=1056
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dropout0=0.6 --l21=0.01 \
                    --batch=32 --dense1=512
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=532 --dropout0=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=150 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=769 --dropout1=0 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --load_model=./output/nasnetmobile-20190326-003401/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=120 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dropout0=0 --l21=0 \
                    --batch=32