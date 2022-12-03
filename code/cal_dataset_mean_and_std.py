import torchvision
import torch
from torch.utils.data import DataLoader
from os import system

#需於任意數據集資料夾底下的其中一組別資料夾內執行

train_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
        ])
#torchvision.datasets.SEMEION
#torchvision.datasets.MNIST
#torchvision.datasets.SVHN
all_data = torchvision.datasets.CIFAR10("../files/", 
                                       download=True,
                                       transform=train_transform)
train_loader = DataLoader(dataset=all_data, batch_size=64, shuffle=True)


def get_mean_std(loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in loader:
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
                num_batches += 1

        mean =  channels_sum/num_batches
        std = (channels_squared_sum/num_batches - mean**2)**0.5

        return mean, std


mean,std = get_mean_std(train_loader)

print("len: ", len(train_loader))
print("mean: ", mean)
print("std: ", std)

system('pause')
