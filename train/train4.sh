# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=10 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.00001 --finetune_epochs2=90 --finetune_steps_per_epoch2=400 \
#                     --freeze=0 --dropout1=0.5 --l21=0.01  \
#                     --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=400 \
#                     --freeze=51 \
#                     --batch=64
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=5 --train_steps_per_epoch=400 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=60 --finetune_steps_per_epoch2=400 \
#                     --freeze=91 \
#                     --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --dense_layers=1 --workers=2 \
                    --train_lr=0.0001 --train_epochs=10 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.00001 --finetune_epochs2=200 --finetune_steps_per_epoch2=400 \
                    --freeze=0 \
                    --batch=32