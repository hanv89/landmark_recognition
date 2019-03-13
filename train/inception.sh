python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=inception_v3 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=249 --dropout=0.5 \
                    --workers=2 --batch=64 --crop=0 --width=0.2 --height=0.2

python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=91 --dropout=0.5 \
                    --workers=2 --batch=64 --crop=0 --width=0.2 --height=0.2 --finetune_lr=0.0002

python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 \
                    --train_epochs=10 --train_steps_per_epoch=1200 \
                    --finetune_epochs=120 --finetune_steps_per_epoch=1200 \
                    --freeze=117 --dropout=0.5 \
                    --workers=2 --batch=64 --crop=0 --width=0.2 --height=0.2 --finetune_lr=0.0004