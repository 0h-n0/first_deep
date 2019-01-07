#!/usr/bin/env python
import sys
import tarfile 
from pathlib import Path


import torch
from urllib.request import urlopen

def get_dataset():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    file_name = Path('cifar-100-python.tar.gz')
    
    res = urlopen(url)
    f = open(file_name, 'wb')
    meta = res.info()
    
    file_size = int(meta["Content-Length"])
    f = tarfile.open(file_name)
    file_size_dl = 0
    
    print(file_name.stat().ST_SIZE)
    
    block_sz = 8192
    while True:
        buffer = res.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = "\rDownloading: %10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        print(status, end='', flush=True)
        

get_dataset()    
