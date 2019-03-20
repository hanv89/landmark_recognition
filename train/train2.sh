# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --workers=2 \
#                     --train_lr=0.0005 --train_epochs=5 --train_steps_per_epoch=600 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=600 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=50 --finetune_steps_per_epoch2=600 \
#                     --freeze=117 --dropout=0 --l2=0 \
#                     --batch=128
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --workers=2 \
#                     --train_lr=0.0005 --train_epochs=5 --train_steps_per_epoch=600 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=600 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=50 --finetune_steps_per_epoch2=600 \
#                     --freeze=532 --dropout=0 --l2=0 \
#                     --batch=128
# python3 finetune_v2.py --data=./data/TrainValAug/ --mode=train_then_finetune --net=densenet_169 --workers=2 \
#                     --train_lr=0.0005 --train_epochs=5 --train_steps_per_epoch=1200 \
#                     --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=1200 \
#                     --finetune_lr2=0.0001 --finetune_epochs2=50 --finetune_steps_per_epoch2=1200 \
#                     --freeze=367 --dropout=0 --l2=0 \
#                     --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=2 --workers=2 \
                    --train_lr=0.0001 --train_epochs=10 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=90 --finetune_steps_per_epoch2=400 \
                    --freeze=91 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=2 --workers=2 \
                    --train_lr=0.0001 --train_epochs=10 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=90 --finetune_steps_per_epoch2=400 \
                    --freeze=51 \
                    --batch=64
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --dense_layers=2 --workers=2 \
                    --train_lr=0.0001 --train_epochs=10 --train_steps_per_epoch=400 \
                    --finetune_lr1=0.0001 --finetune_epochs1=0 --finetune_steps_per_epoch1=400 \
                    --finetune_lr2=0.0001 --finetune_epochs2=90 --finetune_steps_per_epoch2=400 \
                    --freeze=51 --dropout1=0.5 --dropout2=0.5\
                    --batch=64