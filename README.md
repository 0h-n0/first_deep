# first_deep
This repository First se

## requirements

* pytorch >= 1.0.0

## Setup

```shell
$ wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
$ bash ~/miniconda.sh -b -p $HOME/miniconda
$ export PATH="$HOME/miniconda/bin:$PATH"
$ conda update conda
$ conda create -n py36 python=3.6 pip
$ source activate py3
```

install other packages

```shell
$ pip install -r requirements.txt
```

### install pytorch-gpu
```shell
$ conda install pytorch torchvision cuda100 -c pytorch
```

## How to run

### Mnist

```shell
$ python mnist.py
```

### Cifar-100

```shell
$ python cifar.py
```

## Results