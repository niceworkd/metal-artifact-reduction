from torch.utils.data import Dataset,DataLoader
from skimage.io import imread
from skimage.transform import radon,rescale
import numpy as np
import torch
import scipy.io as scio
import h5py
class datas(Dataset):
    def __init__(self,txt_file="./test.txt",root_dir="./new_datas/"):
        self.train_file = txt_file
        self.dic = []
        self.root_dir = root_dir
        f = open(self.train_file,"r")
        for line in f:
            vals = line.split(" ")
            tup = (vals[0].replace('\n',''),vals[1].replace('\n',''))
            self.dic.append(tup)
    def __len__(self):
        return len(self.dic)
    def __getitem__(self, index):
       # data = h5py.File(self.root_dir+self.dic[index][1])
        label = h5py.File(self.root_dir+self.dic[index][0])
        label = np.transpose(label['image'])
        label = label.astype(np.float32)
        
        label = (label-np.min(label)) / (np.max(label)-np.min(label))
        label = label*255
        #data = np.transpose(data['mask_proj'])
        #data = data.astype(np.float32)
        #theta = np.linspace(0.,180,256,endpoint=True)
        #data = radon(label,theta=theta,circle=True)
        #data = data.astype(np.float32)
        label = torch.from_numpy(label)
        #data = torch.from_numpy(data)
        
        #data = data.view(-1,128,256)
        label = label.view(-1,512,512)
        return label,label

