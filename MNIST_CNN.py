import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data  # PyTorch读取数据的一个接口
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 设置超参数
EPOCH = 6  # 把整套数据集训练6遍
BATCH_SIZE = 50  # 每次训练50张图片
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # True表明需要下载MNIST数据集


# 下载训练神经网络的数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 把图片转化为tensor形式
    download=True,
)


