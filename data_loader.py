#!/usr/bin/evn python
# encoding: utf-8

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            imgs.append((int(words[0]), words[1], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        label, rgb_image, depth_image = self.imgs[index]
        rgb_img = cv2.imread(os.path.join('data/images', rgb_image))
        depth_img = cv2.imread(os.path.join(
            'data/images', depth_image), cv2.IMREAD_ANYDEPTH)
        # depth_img = cv2.convertScaleAbs(depth_img,alpha=(255.0/depth_img.max()))
        rgb_img.resize(32, 32, 3)
        depth_img.resize(32, 32)
        rgbd = np.zeros((4, 32, 32), dtype=np.float32)
        rgbd[0, :, :] = rgb_img[:, :,  0]
        rgbd[1, :, :] = rgb_img[:, :,  1]
        rgbd[2, :, :] = rgb_img[:, :,  2]
        rgbd[3, :, :] = depth_img
        if self.transform is not None:
            rgbd = self.transform(rgbd)
        return (rgbd, label)

    def __len__(self):
        return len(self.imgs)


# dataiter = DataLoader(MyDataset(txt_path), batch_size=32, shuffle=True)


if __name__ == "__main__":
    test_data = MyDataset(txt_path='./data/train_list.txt_test')
    print(test_data)
    aa = test_data.__getitem__(0)
    print(aa[0].shape)
    print(test_data.__len__())
