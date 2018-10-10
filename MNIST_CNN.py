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


# 下载MNIST数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 把图片转化为tensor形式
    download=True,
)
#从MNIST中取出测试数据集
test_data = torchvision.datasets.MNIST(
    root='.mnist',
    train=False,
    download=True,
)


# 打印训练数据集的一张图片，看看它图片是怎样的
print('train_data.size: ', train_data.train_labels.size())
print('test_data.size: ', test_data.test_labels.size())
plt.imshow(train_data.train_data[0])  # 展示第一张图
plt.title('%i' % train_data.train_labels[0])
plt.show()


