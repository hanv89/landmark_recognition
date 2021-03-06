# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=400 \
#                     --freeze=51 --dropout1=0.5 --l21=0.01 \
#                     --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=1 --workers=2 \
#                     --train_lr=0.0001 --train_epochs=5 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=400 \
#                     --freeze=139 --dropout1=0.5 --l21=0.1  \
#                     --batch=64

# python3 finetune_v2.py --data=./data/TrainVal/ --mode=finetune --net=densenet_169 --dense_layers=1 --workers=2 \
#                     --load_model=./output/densenet_169-20190321-005731/finetune/sgd/model.h5 \
#                     --train_lr=0.0001 --train_epochs=6 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.00001 --finetune_epochs2=60 --finetune_steps_per_epoch2=400 \
#                     --freeze=139 \
#                     --batch=64

python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=5 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.001 --finetune_epochs2=30 --finetune_steps_per_epoch2=400 \
                    --freeze=595 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=1 --workers=2 \
                    --train_lr=0.0001 --train_epochs=5 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=139 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=inception_v3 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=3 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.001 --finetune_epochs2=8 --finetune_steps_per_epoch2=400 \
                    --freeze=311 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=resnet_50 --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=3 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.001 --finetune_epochs2=8 --finetune_steps_per_epoch2=400 \
                    --freeze=175 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=xception --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=3 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.001 --finetune_epochs2=8 --finetune_steps_per_epoch2=400 \
                    --freeze=132 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=51 --dropout1=0.5 --l21=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=295 --dropout1=0.5 --l21=0 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=80 --finetune_steps_per_epoch2=400 \
                    --freeze=532 \
                    --batch=64