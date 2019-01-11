#!/usr/bin/env python

import torch
import torch.utils.data

from libs.data import PTBDataloaderFactory
from libs.models import RNNModel
from libs.splitcross import SplitCrossEntropyLoss

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Penn Treebank Training Script.')
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

    
def get_dataloaders(
        batch_size=128,
        test_batch_size=1024,
        bptt_length=10,
        num_processes_per_iterator=3,
        shuffle=True):

    PTBDataloaderFactory.set_train_data_path("./data/ptb.train.txt")
    PTBDataloaderFactory.set_test_data_path("./data/ptb.train.txt")
    PTBDataloaderFactory.set_valid_data_path("./data/ptb.train.txt")    
    
    dataloader_factory = PTBDataloaderFactory(batch_size,
                                              test_batch_size,
                                              bptt_length,
                                              num_workers=num_processes_per_iterator,
                                              shuffle=True)

    ntokens = dataloader_factory.ntokens
    train_dataloader = dataloader_factory.get_dataloader('train')
    valid_dataloader = dataloader_factory.get_dataloader('valid')
    test_dataloader = dataloader_factory.get_dataloader('test') 
    
    return train_dataloader, valid_dataloader, test_dataloader, ntokens


def get_model(ntokens, criterion=None):
    emsize = 300
    model = RNNModel(ntokens, 250, emsize, rnn_type="LSTM", num_layers=3)
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
            print('Using', splits)
        criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)
    return model, criterion

def train(epochs, ntokens, train_dataloader, valid_dataloader):
    model, criterion = get_model(ntokens)
    params = list(list(model.parameters()) + list(criterion.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.01)

    model.train()
    
    total_loss = 0
    for epoch in range(epochs):
        for source, targets in train_dataloader:
            _, B, T = source.shape
            source = source.view(B, T)
            targets = targets.view(B, T)
            
            optimizer.zero_grad()                    
            #output, hidden, rnn_hs, dropped_rnn_hs = model(source) #, return_h=True)
            output, hidden = model(source) #, return_h=True)
            output = output.contiguous().view(output.size(0) * output.size(1), output.size(2))
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
            loss = raw_loss
            optimizer.step()
            total_loss += loss.item()
            print('total_loss', total_loss)
        print(epoch)

def test(test_dataloader):
    pass

if __name__ == '__main__':
    train_dataloader, valid_dataloader, test_dataloader, ntokens = get_dataloaders()
    model = get_model(ntokens)
    train(20, ntokens, train_dataloader, valid_dataloader)





