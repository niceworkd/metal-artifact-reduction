import os
import os.path as path
import json
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from random import choice
from torch.utils.data import Dataset
from ..utils import read_dir
import random
import h5py


class DeepLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='data/deep_lesion/train',
        blacklist='data/deep_lesion/blacklist.json', normalize=True, partial_holdout=0,
        hu_offset=32768, random_mask=True, random_flip=False,load_all=True,load_mask=False):
        super(DeepLesion, self).__init__()

        self.prior_dict = {}
        self.sinoLi_dict = {}
        self.trace_dict= {}
        # self.data_dict = {}
        # if blacklist is not None and path.isfile(blacklist):
        #     with open(blacklist) as f: self.blacklist = set(json.load(f))
        # else: self.blacklist = []
        self.blacklist = []

        if type(dataset_dir) is str and path.isdir(dataset_dir):
            # cache_file = path.join(dataset_dir, "cache.json")
            # if path.isfile(cache_file):
            #     with open(cache_file) as f:
            #         self.data_dict = json.load(f)
            # else:
            gt_files = read_dir(
                dataset_dir, predicate=lambda x: x == "Sgt.mat", recursive=True)
            for gt_file in tqdm(gt_files, desc="Create data file list"):
                metal_dir = path.split(gt_file)[0]
                # image_name = "/".join(metal_dir.split(path.sep)[-2:]) + ".png"
                # if image_name in self.blacklist: continue
                # metal_files = sorted(read_dir(metal_dir,
                #     predicate=lambda x: x.endswith("mat") and x != "Sgt.mat"))

                prior_files = sorted(read_dir(metal_dir,
                    predicate=lambda x: x.startswith("SPrior")))
                sinoLi_files = sorted(read_dir(metal_dir,
                    predicate=lambda x: x.startswith("SLi")))
                trace_files = sorted(read_dir(metal_dir,
                    predicate=lambda x: x.startswith("trace")))
                #self.data_dict[gt_file] = [f for f in metal_files]
                self.prior_dict[gt_file] = [f for f in prior_files]
                self.sinoLi_dict[gt_file] = [f for f in sinoLi_files]
                self.trace_dict[gt_file] = [f for f in trace_files]
                # with open(cache_file, 'w') as f:
                #     json.dump(self.data_dict, f)

        self.norm = normalize
        self.random_flip = random_flip
        self.random_mask = random_mask
        self.load_mask = load_mask
        self.partial_holdout = partial_holdout
        self.hu_offset = hu_offset
        #self.data_dict = sorted(self.data_dict.items())
        self.prior_dict = sorted(self.prior_dict.items())
        self.sinoLi_dict = sorted(self.sinoLi_dict.items())
        self.trace_dict = sorted(self.trace_dict.items())

        # if self.partial_holdout:
        #     visible_size = int(len(self.data_dict) * (1 - partial_holdout))
        #     self.visible_dict = self.data_dict[:visible_size]
        #     self.invisible_dict = self.data_dict[visible_size:]
        # else:
        #     self.visible_dict = self.data_dict
        #     self.invisible_dict = []
        # self.visible_files = [(k, f) for k,v in self.visible_dict for f in v]
        # self.invisible_files = [(k, f) for k,v in self.invisible_dict for f in v]

    def to_tensor(self, data, norm=True):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)
        #if norm: data = self.normalize(data, self.get_minmax())
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data, denorm=True):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        #if denorm: data = self.denormalize(data, self.get_minmax())
        return data

    def get_minmax(self):
        return 0.0, 0.5

    def convert2coefficient(self, image):
        # MIUWATER = 0.192

        # image = np.array(image, dtype=np.float32)
        # image = image - self.hu_offset
        # image[image < -1000] = -1000
        # image = image / 1000 * MIUWATER + MIUWATER

        return image

    def load_data(self, data_file):
        gt = h5py.File(data_file[0])['image'][:]
        #gt = sio.loadmat(data_file[0])['image']
        #metal = sio.loadmat(data_file[1])['image']
        prior = h5py.File(data_file[1])['image'][:]
        sinoLi = h5py.File(data_file[2])['image'][:]
        trace = h5py.File(data_file[3])['image'][:]
        return gt,prior,sinoLi,trace

    def normalize(self, data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 2.0 - 1.0
        return data

    def denormalize(self, data, minmax):
        data_min, data_max = minmax
        data = data * 0.5 + 0.5
        data = data * (data_max - data_min) + data_min
        return data

    def get(self, data_file):
        data_name = data_file[1]

        # load images
        #gt, metal = self.load_data(data_file)
        gt,prior,sinoLi,trace = self.load_data(data_file)
        # if self.partial_holdout:
        #     gt, _ = self.load_data(choice(self.invisible_files))

        # if self.random_flip and np.random.rand() >= 0.5:
        #     gt, metal = gt[::-1, :], metal[::-1, :]

        # if self.load_mask:
        #     vmax = self.get_minmax()[-1]
        #     mask = (metal > vmax).astype(np.float32)

        gt, prior,sinoLi,trace = self.to_tensor(gt), self.to_tensor(prior),self.to_tensor(sinoLi),self.to_tensor(trace)
        data = {"data_name": data_name, "gt": gt, "prior": prior, "li":sinoLi, "trace":trace}
        return data

    def __len__(self):
        # if self.random_mask: return len(self.visible_dict)
        # else: return len(self.visible_files)
        return len(self.prior_dict)
    def __getitem__(self, index):
        # if self.random_mask:
        #     gt_file, metal_files = self.visible_dict[index]
        #     data_file = gt_file, choice(metal_files)
        # else:
        #     data_file = self.visible_files[index]
        gt_file,prior_files = self.prior_dict[index]
        gt_file,sinoLi_files = self.sinoLi_dict[index]
        gt_file,trace_files = self.trace_dict[index]
        offset = random.randint(0,89)
        data_file = gt_file,prior_files[offset],sinoLi_files[offset],trace_files[offset]
        return self.get(data_file)
