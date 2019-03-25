python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=2 --workers=2 \
                    --load_model=./output/nasnetmobile-20190323-170225/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0 --l21=0.01 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=2 --workers=2 \
                    --load_model=./output/nasnetmobile-20190323-100616/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0.2 --l21=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=nasnetmobile --dense_layers=2 --workers=2 \
                    --load_model=./output/nasnetmobile-20190323-013106/finetune/sgd/model.h5 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0002 --finetune_epochs1=10 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0002 --finetune_epochs2=40 --finetune_steps_per_epoch2=400 \
                    --freeze=295 \
                    --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.0002 --finetune_epochs2=120 --finetune_steps_per_epoch2=400 \
#                     --freeze=55 --dropout1=0 --l21=0 \
#                     --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=100 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=100 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=100 \
#                     --freeze=295 --dropout1=0 --l21=0 \
#                     --batch=256
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=200 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=200 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=200 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=128
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=800 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=32
