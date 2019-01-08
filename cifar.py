#!/usr/bin/env python
import sys
import tarfile
import pickle
from pathlib import Path

import torch
import torch.utils.data
from urllib.request import urlopen

import libs.autoaugment
import libs.models

class Dataset(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.size
        

class Preprocess(object):
    def __init__(self):
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.file_name = Path('cifar-100-python.tar.gz')
        self.data_dir = Path('cifar-100-python')
        self.get_dataset()
        self.train_dataloaer, self.test_dataloader = self.get_dataloader()
        
    def get_dataset(self) -> bool:
        if self.file_name.exists():    
            try:
                t = tarfile.open(self.file_name, 'r:gz')
                print("Skip: Downloading.")            
                return True
            except tarfile.ReadError as e:
                print(f"{file_name} is broken.")
                print("Re-Downloading:...")
            
        res = urlopen(self.url)
        f = self.file_name.open('wb')
        meta = res.info()
        
        file_size = int(meta["Content-Length"])
        file_size_dl = 0
        
        block_sz = 8192
        while True:
            buffer = res.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = "\rDownloading: %10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            print(status, end='', flush=True)
            print('Finished:')        
        return True

    def _read_targz(self):
        tar = tarfile.open(self.file_name, "r:gz")
        tar.extractall()
        train_dict = self.unpickle(self.data_dir / 'train')
        test_dict = self.unpickle(self.data_dir / 'test')
        return train_dict, test_dict

    def unpickle(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data
    
    def get_dataloader(self):
        train_dict, test_dict = self._read_targz()
        train_data = train_dict[b'data']
        train_fine_labels = train_dict[b'fine_labels']
        
        test_data = test_dict[b'data']
        test_fine_labels = test_dict[b'fine_labels']

        train_dataset = Dataset(train_data, train_fine_labels)
        test_dataset = Dataset(test_data, test_fine_labels)
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_dataset)

        return train_dataloader, test_dataloader


class Trainer(object):
    def __init__(self, data_loader,):
        pass

def train():
    pass

def test():
    pass

def main():
    train()
    test()



if __name__ == '__main__':
    
    p = Preprocess()
    main()
