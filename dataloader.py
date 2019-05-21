import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import wavfile
import random

class LoadDataset(Dataset):
    def __init__(self, ds_path, mode, resize_shape):
        if mode == "train":
            f_path = "ASVspoof2017_V2_train"
        elif mode == "dev":
            f_path = "ASVspoof2017_V2_dev"
        elif mode == "eval":
            f_path = "ASVspoof2017_V2_eval"
        else:
            raise ValueError('Incorrect mode')
            
        self.ds_path = ds_path + f_path
        if mode == "train":
            self.files_p = ds_path + f_path + ".trn.txt"
        else:
            self.files_p = ds_path + f_path + ".trl.txt"
        self.sample_size = 10752
        with open(self.files_p, 'r') as f:
            data = f.read().splitlines()

        self.files = [os.path.join(self.ds_path, x.split()[0]) for x in data]
        self.targets = [int(x.split()[1] == "genuine") for x in data]


    def __getitem__(self, index):
        im, lb = self.pull_item(index)
        return im, lb

    def __len__(self):
        return len(self.files)

    def pull_item(self, index):

        _, x = wavfile.read(self.files[index])
        x = np.asarray(x, dtype='float32')
        x /= np.max(np.abs(x))

        # Convert from NumPy array to PyTorch Tensor.
        x = torch.from_numpy(x)

        # If it's too long, truncate it.
        if x.numel() > self.sample_size:
            samples_number = x.numel() // self.sample_size #nubmer of samples per signal
        n_sample = random.randint(0, samples_number-1)


        sample = x[n_sample*self.sample_size:n_sample*self.sample_size + self.sample_size]
        return sample.float().cuda(), torch.LongTensor([self.targets[index]]).cuda()
