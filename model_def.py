import torch
import torch.nn as nn
import params
import torchvision
import torch.nn.functional as F
from torchvision.datasets import Flowers102
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy
from torch.nn.utils.rnn import pad_sequence


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
        self.bat1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bat2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bat3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv4 = nn.Conv2d(128, 306, 1)
        self.bat4 = nn.BatchNorm1d(306)
        #self.fc1 = nn.Linear(3200, 306)
        self.fc2 = nn.Linear(306, 306)
        self.fc4 = nn.Linear(306, 102)

    def forward(self, x):
        x = self.pool(F.relu(self.bat1(self.drop(self.conv1(x)))))
        x = self.pool(F.relu(self.bat2(self.drop(self.conv2(x)))))
        x = self.pool(F.relu(self.bat3(self.drop(self.conv3(x)))))
        x = self.conv4(x)
        #print(x.shape)
        x = F.max_pool2d(x, kernel_size = x.size()[2:])
        #print(x.shape)
        x = x.view(-1, 306)
        #print(x.shape)
        x = F.relu(self.bat4(self.fc2(x)))
        #print(x.shape)
        x = self.fc4(x)
        return x


def save_model(model_to_save):
    print("Saving Model")
    torch.save(model_to_save[0].state_dict(), params.model_path_save)


def load_model():
    print("Loading Model")
    model = CNN()
    model.load_state_dict(torch.load(params.model_path_load))
    return model
