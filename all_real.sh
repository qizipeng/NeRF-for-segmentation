#!/bin/bash

start="start!!"
echo $start

echo "if train and test exist then delete "
if [ -d "./data/carla/carla/train" ];then
rm -rf ./data/carla/carla/train
fi
if [ -d "./data/carla/carla/test" ];then
rm -rf ./data/carla/carla/test
fi
cp -rf ./data/carla/carla/real1/train ./data/carla/carla/
cp -rf ./data/carla/carla/real1/test ./data/carla/carla/
echo "train real1"
python train.py --config=./configs/carlareal1.txt
echo "test real1"
python test.py --config=./configs/carlareal1.txt

echo "if train and test exist then delete "
if [ -d "./data/carla/carla/train" ];then
rm -rf ./data/carla/carla/train
fi
if [ -d "./data/carla/carla/test" ];then
rm -rf ./data/carla/carla/test
fi
cp -rf ./data/carla/carla/real2/train ./data/carla/carla/
cp -rf ./data/carla/carla/real2/test ./data/carla/carla/
echo "train real2"
python train.py --config=./configs/carlareal2.txt
echo "test real2"
python test.py --config=./configs/carlareal2.txt

echo "if train and test exist then delete "
if [ -d "./data/carla/carla/train" ];then
rm -rf ./data/carla/carla/train
fi
if [ -d "./data/carla/carla/test" ];then
rm -rf ./data/carla/carla/test
fi
cp -rf ./data/carla/carla/real3/train ./data/carla/carla/
cp -rf ./data/carla/carla/real3/test ./data/carla/carla/
echo "train real3"
python train.py --config=./configs/carlareal3.txt
echo "test real3"
python test.py --config=./configs/carlareal3.txt

echo "finish realdata"


