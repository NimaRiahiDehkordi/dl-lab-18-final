#!/bin/sh
END=10

for ((i=1;i<=END;i++)); do
    python3 ./intensity/Train_Net.py --checkpoint_dir=./intensity/checkpoints/$i/ --configuration=4 --logs_path=./intensity/logs/$i/ --dataset='CamVidV300'
    python3 ./intensity/Test_Net.py --model_path=./intensity/checkpoints/$i --configuration=4 --dataset='CamVidV300'
done

