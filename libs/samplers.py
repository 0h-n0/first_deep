import math

from torch.utils.data.sampler import Sampler

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
