import torch
import torch.nn as nn
import torchvision
import os
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

import dataOperations
import model_def
import params


def test_on_batch():

    tl = test_loader
    # get batch of images from the test DataLoader
    images, labels = next(iter(tl))

    # show all images as one image grid
    dataOperations.show_img(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(params.batch_size)))

    # Let's see what if the model identifiers the  labels of those example

    images = images.to(params.device)
    labels = labels.to(params.device)
    outputs = mod1(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(params.batch_size)))


def find_acc_on_all_classes(isTrain):
    if isTrain:
        selected_loader = train_loader
    else:
        selected_loader = test_loader
    class_correct = list(0. for i in range(len(params.flowerNames)))
    class_total = list(0. for i in range(len(params.flowerNames)))
    with torch.no_grad():
        for data in selected_loader:
            images, labels = data
            images = images.to(params.device)
            labels = labels.to(params.device)
            outputs = mod1(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(params.flowerNames)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def test_accuracy(isTrain):
    if isTrain:
        selected_loader = train_loader
    else:
        selected_loader = test_loader
    mod1.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for data in selected_loader:
            images, labels = data
            images, labels = images.to(params.device), labels.to(params.device)
            # run the model on the test set to predict labels
            outputs = mod1(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100.0 * float(accuracy) / float(total))
    return float(accuracy)


def train(num_epochs):
    best_accuracy = 0.0
    ctr = 0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = images.to(params.device)
            labels = labels.to(params.device)

            # zero the parameter gradients
            optim.zero_grad()
            # predict classes using images from the training set
            outputs = mod1(images)
            # print("Outputs shape : %s, Labels shape %s" % (outputs.shape, labels.shape))
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optim.step()

            running_loss += loss.item()  # extract the loss value
            if i % 99 == 0 and i != 0:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 99))
                # zero the loss
                running_loss = 0.0

        if ctr==1 or ctr ==2 or ctr == 10 or ctr == 20 or ctr == 30 or ctr == 35 or ctr == 36 or ctr == 37 or ctr == 38 or ctr == 39:
            acctrain = test_accuracy(True)
            acctest = test_accuracy(False)
            print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %s %%' % acctest)
            print('For epoch', epoch + 1, 'the test accuracy over the whole train set is %s %%' % acctrain)

            # we want to save the model if the accuracy is the best
            if acctest > best_accuracy:
                model_def.save_model([mod1])
                best_accuracy = acctest
        ctr += 1

def print_mem_usage():
    print("Tot %s" % torch.cuda.get_device_properties(0).total_memory)
    print("Reserve %s" % torch.cuda.memory_reserved(0))
    print("Alloc %s" %torch.cuda.memory_allocated(0))

trf1 = transforms.Compose([transforms.RandomRotation(360),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip(),
                           transforms.ToTensor(),
                           dataOperations.SquarePadTo1000(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

tr_test_set = transforms.Compose([transforms.ToTensor(),
                                  dataOperations.SquarePadTo1000(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ])



train_set = Flowers102(root="./data", split="train", transform=trf1, download=True)
test_set = Flowers102(root="./data", split="test", transform=tr_test_set, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False,
                                          num_workers=0)




if __name__ == "__main__":


    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    print("The number of images in a training set is: ", len(train_loader) * params.batch_size)
    print("The number of images in a test set is: ", len(test_loader) * params.batch_size)
    print("The number of batches per epoch is: ", len(train_loader))

    classes = params.flowerNames
    if params.train_mode:

        mod1 = model_def.CNN()
        optim = torch.optim.Adam(mod1.parameters(), lr=params.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        mod1 = mod1.to(params.device)
        train(params.amount_epochs)



    mod1 = model_def.load_model()
    mod1.to(params.device)
    acctrload= test_accuracy(True)
    acctstload = test_accuracy(False)
    print('The train accuracy of the loaded model is %s ' % acctrload)
    print('The test accuracy of the loaded model is %s' % acctstload)

    test_on_batch()

    find_acc_on_all_classes(True)
    print("////////////////////////////")
    find_acc_on_all_classes(False)


