import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class TrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()

        self.dataset = os.path.join(opt.train_dataset, 'imagenet.txt')
        self.image_path = opt.train_dataset
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = self.image_path+file_name.strip()
        in_img = cv2.imread(img_file)

        inp_img = Image.fromarray(in_img)

        ps = opt.image_size
        w, h = inp_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')


        inp_img = TF.to_tensor(inp_img)

        hh, ww = inp_img.shape[1], inp_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]

        #=======================

        c_img  = np.zeros((256,256,3),np.uint8)
        c_img[:] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        c_img = Image.fromarray(c_img)
        c_img = TF.to_tensor(c_img)

        # inp_img = inp_img * 0.5 + c_img * 0.5

        sample = {'in_img': inp_img, 'c_img': c_img}

        return sample




