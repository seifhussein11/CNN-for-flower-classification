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

import runTraining


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


def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


def train(num_epochs):
    best_accuracy = 0.0
    ctr = 0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # get the inputs
            images = images.to(device)
            labels = labels.to(device)
            runTraining.print_mem_usage()

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # print("Outputs shape : %s, Labels shape %s" % (outputs.shape, labels.shape))
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item()  # extract the loss value
            if i % 99 == 0 and i != 0:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 99))
                # zero the loss
                running_loss = 0.0

        if ctr==1 or ctr ==2 or ctr == 10 or ctr == 20 or ctr == 30 or ctr == 35 or ctr == 36 or ctr == 37 or ctr == 38 or ctr == 39:
            accuracy = testAccuracy()
            accuracy2 = testAccTrain()
            print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %s %%' % accuracy)
            print('For epoch', epoch + 1, 'the test accuracy over the whole test train is %s %%' % accuracy2)

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                saveModel()
                best_accuracy = accuracy
        ctr += 1


def testClassess():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            model.cuda(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                try :
                    class_correct[label] += c[i].item()
                except IndexError:
                    class_correct[label] += c.item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def testClassessTrainData():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            model.cuda(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                try :
                    class_correct[label] += c[i].item()
                except IndexError:
                    class_correct[label] += c.item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def Find_Largest_Size():
    tr1 = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_setOG2 = Flowers102(root="./data", split="train", transform = tr1, download=True)

    train_loader_find_size = torch.utils.data.DataLoader(train_setOG2, batch_size=1, shuffle=True,
                                               num_workers=0)

    smallestx = 0
    smallesty = 0
    for data in train_loader_find_size:
        image, label = data
        _, _, x, y = image.size()
        if x > smallestx:
            smallestx = x
        if y > smallesty:
            smallesty = y
    print(smallestx, smallesty)


class SquarePad:
    def __call__(self, image):
        x, h, w = image.size()
        max_wh = 1000
        padl = int((max_wh - w) / 2)
        padt = int((max_wh - h) / 2)

        padr = max_wh - w - padl
        padb = max_wh - h - padt
        padding = (padl, padr, padt, padb)
        return F.pad(image, padding, 'constant', 0)

def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100.0 * float(accuracy) / float(total))
    return float(accuracy)

def testAccTrain():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100.0 * float(accuracy) / float(total))
    return float(accuracy)



def imageshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))


flowerNames = [
    'pink primrose',
 'hard-leaved pocket orchid',
 'canterbury bells',
 'sweet pea',
 'english marigold',
 'tiger lily',
 'moon orchid',
 'bird of paradise',
 'monkshood',
 'globe thistle',
 'snapdragon',
 "colt's foot",
 'king protea',
 'spear thistle',
 'yellow iris',
 'globe-flower',
 'purple coneflower',
 'peruvian lily',
 'balloon flower',
 'giant white arum lily',
 'fire lily',
 'pincushion flower',
 'fritillary',
 'red ginger',
 'grape hyacinth',
 'corn poppy',
 'prince of wales feathers',
 'stemless gentian',
 'artichoke',
 'sweet william',
 'carnation',
 'garden phlox',
 'love in the mist',
 'mexican aster',
 'alpine sea holly',
 'ruby-lipped cattleya',
 'cape flower',
 'great masterwort',
 'siam tulip',
 'lenten rose',
 'barbeton daisy',
 'daffodil',
 'sword lily',
 'poinsettia',
 'bolero deep blue',
 'wallflower',
 'marigold',
 'buttercup',
 'oxeye daisy',
 'common dandelion',
 'petunia',
 'wild pansy',
 'primula',
 'sunflower',
 'pelargonium',
 'bishop of llandaff',
 'gaura',
 'geranium',
 'orange dahlia',
 'pink-yellow dahlia?',
 'cautleya spicata',
 'japanese anemone',
 'black-eyed susan',
 'silverbush',
 'californian poppy',
 'osteospermum',
 'spring crocus',
 'bearded iris',
 'windflower',
 'tree poppy',
 'gazania',
 'azalea',
 'water lily',
 'rose',
 'thorn apple',
 'morning glory',
 'passion flower',
 'lotus',
 'toad lily',
 'anthurium',
 'frangipani',
 'clematis',
 'hibiscus',
 'columbine',
 'desert-rose',
 'tree mallow',
 'magnolia',
 'cyclamen ',
 'watercress',
 'canna lily',
 'hippeastrum ',
 'bee balm',
 'ball moss',
 'foxglove',
 'bougainvillea',
 'camellia',
 'mallow',
 'mexican petunia',
 'bromelia',
 'blanket flower',
 'trumpet creeper',
 'blackberry lily']



transformations = transforms.Compose([
                                      transforms.RandomRotation(360),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      SquarePad(),
                                      ])


transformationsNoChange = transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              SquarePad()
                                              ])

transformsTestSet = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        SquarePad(),
                                        ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
print(len(flowerNames))
number_of_labels = 102


train_setOG = Flowers102(root="./data", split="train", transform=transformations, download=True)


print("loading train")
train_loader = torch.utils.data.DataLoader(train_setOG, batch_size=batch_size, shuffle=True,
                                           num_workers=0)
print("end load train")

test_set = Flowers102(root="./data", split="test", transform=transformsTestSet, download=True)

print("loading test")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                          num_workers=0)
print("end load test")

print("The number of images in a training set is: ", len(train_loader)*batch_size)
print("The number of images in a test set is: ", len(test_loader)*batch_size)
print("The number of batches per epoch is: ", len(train_loader))

classes = flowerNames
model = CNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")
# Convert model parameters and buffers to CPU or Cuda
model.to(device)


if __name__ == "__main__":
    Find_Largest_Size()
    # Let's build our model
    amount = 40
    train(amount)
    print('Finished Training')

    # Let's load the model we just created and test the accuracy per label
    model = CNN()
    path = "./myFirstModel.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    # Test with batch of images
    testBatch()

    testClassessTrainData()

    print("///////////////////////////////////")

    testClassess()



