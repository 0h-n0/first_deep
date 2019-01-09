# first_deep

This repogitory provides a first step for learning Deep Learning method and mainly treats Cifar-100 Dataset which is a little hard to train a deep neural net model.

## requirements

* pytorch >= 1.0.0
* torchex >= 0.0.4

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