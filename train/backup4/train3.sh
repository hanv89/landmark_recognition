python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=1600 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=1600 \
                    --finetune_lr2=0.001 --finetune_epochs2=80 --finetune_steps_per_epoch2=1600 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=16
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=3200 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=3200 \
                    --finetune_lr2=0.001 --finetune_epochs2=80 --finetune_steps_per_epoch2=3200 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=8
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=6400 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=6400 \
                    --finetune_lr2=0.001 --finetune_epochs2=80 --finetune_steps_per_epoch2=6400 \
                    --freeze=295 --dropout1=0 --l21=0 \
                    --batch=4
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=3200 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=12800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=80 --finetune_steps_per_epoch2=12800 \
#                     --freeze=295 --dropout1=0 --l21=0 \
#                     --batch=2