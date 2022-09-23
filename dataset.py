import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device('cuda:0')

#### 数据类
class custmDataSet(Dataset):
    def __init__(self,root_path, img_dir, H=512,W=512):
        # super(MyDataSet, self).__init__()
        self.root_path = root_path
        self.img_dir = img_dir
        self.img_paths = os.listdir(os.path.join(root_path,img_dir))
        self.H = H
        self.W = W
        self.compose = transforms.Compose(
            [
                transforms.Resize([self.H, self.W]),
                transforms.ToTensor(),
            ]
        )
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path,self.img_dir, self.img_paths[idx])
        img = Image.open(img_path)
        img = self.compose(img)
        return img


    def __len__(self):
        return len(self.img_paths)


class NerfDataSet(Dataset):
    def __init__(self,root_path, split, H=512,W=512):
        super(NerfDataSet, self).__init__()
        self.root_path = root_path
        self.split = split
        self.rgb = os.path.join(self.root_path, self.split,"rgb")
        self.intrinsics = os.path.join(self.root_path, self.split, "intrinscis")
        self.pose = os.path.join(self.root_path, self.split, "pose")




        self.img_paths = os.listdir(os.path.join(root_path,img_dir))
        self.H = H
        self.W = W


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path,self.img_dir, self.img_paths[idx])
        img = Image.open(img_path)
        img = self.compose(img)
        return img


    def __len__(self):
        return len(self.img_paths)

    def sample_random_points(self):
        pass
    def sample_random_patchs(self):
        pass




if __name__ == "__main__":

    root_path = "./data"
    trainorval = "train"
    dataset = MyDataSet(root_path,trainorval)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    print(len(train_loader))






