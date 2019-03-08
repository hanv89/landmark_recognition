# python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=densenet_169 --train_epochs=20 --train_steps_per_epoch=2000 --finetune_epochs=200 --finetune_steps_per_epoch=2000 --freeze=139
# python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=xception --train_epochs=20 --train_steps_per_epoch=2000 --finetune_epochs=200 --finetune_steps_per_epoch=2000 --freeze=56
# 
# python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=mobilenet_v2 --train_epochs=20 --train_steps_per_epoch=2000 --finetune_epochs=200 --finetune_steps_per_epoch=2000 --freeze=73
# python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetlarge --train_epochs=20 --train_steps_per_epoch=2000 --finetune_epochs=200 --finetune_steps_per_epoch=2000 --freeze=0
# python3 finetune.py --data=./data/TrainVal/ --mode=finetune --net=xception --finetune_epochs=500 --finetune_steps_per_epoch=2000 --freeze=56 --load_model=output/xception-20190224-090007/finetune/model.h5


python3 finetune.py --data=./data/TrainVal/ --mode=train_then_finetune --net=nasnetmobile --train_epochs=30 --train_steps_per_epoch=2000 --finetune_epochs=200 --finetune_steps_per_epoch=2000 --freeze=354 --workers=2  --dropout=0.5