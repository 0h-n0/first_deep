import math
from pathlib import Path

import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from .reserved_tokens import UNKNOWN_TOKEN
from .reserved_tokens import EOS_TOKEN
from .encoders import IdentityEncoder


class BPTTSampler(Sampler):
    def __init__(self, data, bptt_length, type_='source'):
        self.data = data
        self.bptt_length = bptt_length
        self.type = type_

    def __iter__(self):
        for i in range(0, len(self.data) - 1, self.bptt_length):
            seq_length = min(self.bptt_length, len(self.data) - 1 - i)
            if self.type == 'source':
                yield slice(i, i + seq_length)
            if self.type == 'target':
                yield slice(i + 1, i + 1 + seq_length)

    def __len__(self):
        return math.ceil((len(self.data) - 1) / self.bptt_length)
    


class BPTTBatchSampler(object):
    def __init__(self, data, bptt_length, batch_size, drop_last, type_='source'):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        chunk_sizes = [math.floor(len(data) / batch_size)] * batch_size

        # Distribute the remaining elements to some chunks
        if not self.drop_last:
            remainder = len(data) - sum(chunk_sizes)
            for i in range(remainder):
                chunk_sizes[i] += 1

        self.samplers = [{
            'offset': sum(chunk_sizes[:i]),
            'sampler': BPTTSampler(range(chunk_sizes[i]), bptt_length, type_=type_)
        } for i in range(batch_size)]

    def __iter__(self):
        self.iterators = [iter(value['sampler']) for value in self.samplers]
        while True:
            batch = []
            for i, iterator in enumerate(self.iterators):
                try:
                    # Adjust the sampler indices to the offset
                    offset = self.samplers[i]['offset']
                    slice_ = next(iterator)
                    batch.append(slice(slice_.start + offset, slice_.stop + offset))
                except StopIteration:
                    pass

            # Samplers are all empty
            if (len(batch) == 0):
                break

            yield batch

    def __len__(self):
        return len(self.samplers[0]['sampler'])


class DummyDataset(Dataset):
    def __init__(self, data_source, source_sampler, target_sampler):
        self.data_source = data_source
        self.source_sampler = list(source_sampler)
        self.target_sampler = list(target_sampler)
        self.size = len(self.source_sampler)

    def __getitem__(self, idx):
        data = torch.stack([self.data_source[i] for i in self.source_sampler[idx]])
        targets = torch.stack([self.data_source[i] for i in self.target_sampler[idx]]).view(-1)
        
        return data, targets

    def __len__(self):
        return self.size


class PTBDataloaderFactory(object):
    train_data_path = ''
    train_test_path = ''
    train_valid_path = ''
    
    @classmethod
    def set_train_data_path(cls, path):
        cls.train_data_path = path

    @classmethod        
    def set_test_data_path(cls, path):
        cls.test_data_path = path

    @classmethod        
    def set_valid_data_path(cls, path):
        cls.valid_data_path = path
    
    def __init__(self, batch_size=128, test_batch_size=1024, bptt_length=10, num_workers=3, shuffle=True):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size        
        self.bptt_length = bptt_length
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.raw_data = dict(
            train=self._preprocess(self.train_data_path),
            valid=self._preprocess(self.valid_data_path),
            test=self._preprocess(self.test_data_path))
        
        self.encoder = IdentityEncoder(self.raw_data['train'] +
                                       self.raw_data['valid'] +
                                       self.raw_data['test'])

        self.ntokens = self.encoder.vocab_size
        self.data = dict(
            train=self.encoder.encode(self.raw_data['train']),
            valid=self.encoder.encode(self.raw_data['valid']),
            test=self.encoder.encode(self.raw_data['test']))

    def get_dataloader(self, mode='train'):
        source_sampler = self._sampler(self.data[mode], self.bptt_length, self.batch_size, 'source')
        target_sampler = self._sampler(self.data[mode], self.bptt_length, self.batch_size, 'target')
        dataset = DummyDataset(self.data[mode], source_sampler, target_sampler)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=self.shuffle,
                                                 num_workers=self.num_workers)
        return dataloader

    def _sampler(self, text, bptt_length, batch_size, type_):
        return BPTTBatchSampler(text, bptt_length, batch_size, True, type_)

    def _preprocess(self, path):
        full_path = Path(path).expanduser().resolve()
        text = []
        with full_path.open(encoding='utf-8') as f:
            for line in f:
                text.extend(line.replace('<unk>', UNKNOWN_TOKEN).split())
                text.append(EOS_TOKEN)
        return text

