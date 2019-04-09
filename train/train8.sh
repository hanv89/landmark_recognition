python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dense1=1024 --dropout0=0.6 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dense1=1024 --dropout0=0.7 --l21=0 \
                    --batch=32
python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=2 --workers=2 \
                    --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
                    --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
                    --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1\
                    --freeze=295 --dropout0=0.5 --dropout1=0.5 --l21=0 \
                    --batch=32 --dense1=1024 --dense2=1024
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1 \
#                     --freeze=295 --dropout0=0.5 --l21=0 \
#                     --batch=32 --validation_split=0.2 --seed=1
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1 \
#                     --freeze=295 --dropout0=0.5 --l21=0 \
#                     --batch=32 --validation_split=0.2 --seed=2
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1 \
#                     --freeze=295 --dropout0=0.5 --l21=0 \
#                     --batch=32 --validation_split=0.2 --seed=3
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1 \
#                     --freeze=295 --dropout0=0.5 --l21=0 \
#                     --batch=32 --validation_split=0.2 --seed=4
# python3 finetune_v2.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --dense_layers=1 --workers=2 \
#                     --train_lr=0.001 --train_epochs=6 --train_steps_per_epoch=800 \
#                     --finetune_lr1=0.001 --finetune_epochs1=0 --finetune_steps_per_epoch1=800 \
#                     --finetune_lr2=0.001 --finetune_epochs2=200 --finetune_steps_per_epoch2=800 --finetune_lr_decay=0.1 \
#                     --freeze=295 --dropout0=0.5 --l21=0 \
#                     --batch=32 --validation_split=0.2 --seed=5