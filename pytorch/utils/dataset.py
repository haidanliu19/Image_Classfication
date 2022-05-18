# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

def cifar10_datast(num_workers, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    batch_size = batch_size
    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, 
                                             download = True, transform = transform)    
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, 
                                             download = True, transform = transform)    
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes