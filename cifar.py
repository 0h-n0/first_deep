#!/usr/bin/env python
import sys
import pickle
import tarfile
import argparse
from pathlib import Path


import torch
import torch.utils.data
from urllib.request import urlopen

import libs.augmentaion as augmentaion
import libs.models as models

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
    def __init__(self, config: Config):
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.file_name = Path('cifar-100-python.tar.gz')
        self.data_dir = Path('cifar-100-python')
        self.get_dataset()
        self.config = config
        
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
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.config.batch_size,
                                                       shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.config.batch_size,
                                                      shuffle=True)

        return train_dataloader, test_dataloader


class Trainer(object):
    def __init__(self, model, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _run(self, data_loader, train=True):
        for x, t in data_loader:
            print(x, t)
        
    def run(self):
        for i in range(self.config.epochs):
            self._run(self.train_loader, train=True)
        self._run(self.test_loader, train=True)
        


class Config(object):
    def __init__(self, epochs, batch_size, test_batch_size, no_cuda, lr,
                 momentum, seed, log_interval, save_model, model_name, augument_type):
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size        
        self.use_cuda = not no_cuda and torch.cuda.is_available()
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.log_interval = log_interval
        self.save_model = save_model

        self.model_name = model_name
        self.augument_type = augument_type
        
    @classmethod
    def get_config(cls, args):
        config = cls(args.epochs, args.batch_size, args.test_batch_size,
                     args.no_cuda, args.lr, args.momentum, args.seed,
                     args.log_interval, args.save_model, args.model_name,
                     args.augument_type)
        return config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cifar-100 Training Script.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--model-name', type=str, default='CNN',
                        choices=['CNN'],
                        help='set a training model.')
    parser.add_argument('--augument-type', type=str, default="None",
                        choices=['None'],                        
                        help='set datay arugumentation type.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    p = Preprocess()
    config = Config.get_config(args)    
    model = models.CNN()
    train_dataloader, test_dataloader = p.get_dataloader()
    t = Trainer(model, config, train_dataloader, test_dataloader)
    t.run()


if __name__ == '__main__':
    main()
