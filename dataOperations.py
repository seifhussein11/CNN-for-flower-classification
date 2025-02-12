import torch
import torch.nn as nn
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

import params


class SquarePadTo1000:
    def __call__(self, image):
        d, h, w = image.size()
        max_wh = 1000
        padl = int((max_wh - w) / 2)
        padt = int((max_wh - h) / 2)
        padr = max_wh - w - padl
        padb = max_wh - h - padt
        padding = (padl, padr, padt, padb)
        return F.pad(image, padding, 'constant', 0)


def show_img(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def find_largest_x_and_y_dimensions():
    tr1 = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set_no_transforms = Flowers102(root="./data", split="train", transform = tr1, download=True)

    train_loader_find_size = torch.utils.data.DataLoader(train_set_no_transforms, batch_size=1, shuffle=True,
                                                         num_workers=0)

    largest_x = 0
    largest_y = 0
    for data in train_loader_find_size:
        image, label = data
        _, _, x, y = image.size()
        if x > largest_x:
            largest_x = x
        if y > largest_y:
            largest_y = y
    return largest_x, largest_y

