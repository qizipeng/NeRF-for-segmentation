tart="start!!"
echo $start

echo "if train and test exist then delete "
if [ -d "./data/carla/carla/train" ];then
	rm -rf ./data/carla/carla/train
fi
if [ -d "./data/carla/carla/test" ];then
	rm -rf ./data/carla/carla/test
fi
cp -rf ./data/carla/carla/8/train ./data/carla/carla/
cp -rf ./data/carla/carla/8/test ./data/carla/carla/
#echo "train real1"
#python train.py --config=./configs/carlareal1.txt
echo "test 8"
python test.py --config=./configs/carla8.txt
