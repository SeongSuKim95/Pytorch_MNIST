import numpy as np
import torch
import torchvision
import cv2
from torchvision import datasets,transforms
from torch.utils.data import dataloader

data_dir = 'mnist'
train_transform = transforms.Compose([transforms.ToTensor()])
data_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)

# data_split
train, test = torch.utils.data.random_split(data_set,[55000,5000])
#
#torchvision.datasetes.MNIST.data_set
print(data_set.data.size()) # torch.Size([60000,28,28])
print(list(data_set.data.size()))

MNIST_np = data_set.data.numpy()

MNIST_mean = np.mean(MNIST_np,axis=0)
dst = np.zeros_like(MNIST_mean)
MNIST_normalized = cv2.normalize(MNIST_mean,dst,0,255,cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('MNIST_mean',MNIST_normalized)
cv2.waitKey(0)
print(MNIST_normalized)

# MNIST mean and var
# 0-1 Normalized mean -> 0.1307
print(data_set.data.float().mean()/255)

# 0-1 Normalized standard deviation -> 0.3081
print(data_set.data.float().std()/255)

#Cifar 10
data_dir = 'cifar10'
train_transform = transforms.Compose([transforms.ToTensor()])
data_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

print(type(data_set.data))  #data type check -> Numpy

CIFAR10 = data_set.data
print(data_set.data.size)
print(data_set.data.shape)

CIFAR10_mean = np.mean(CIFAR10,axis=0).astype(np.uint8)

print(CIFAR10_mean.shape)
cv2.imshow('CIFAR10 mean',CIFAR10_mean)
cv2.waitKey(0)

#CIFAR10 Mean and std
# 0-1 Normalized mean -> 0.473
print(np.mean(data_set.data).astype(np.float)/255)

# 0-1 Normalized standard deviation ->0.251
print(np.std(data_set.data).astype(np.float)/255)
