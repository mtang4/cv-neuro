import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import os
import random
import argparse

from fmri import fmri
from transforms import ChannelSelect
from datanet import complete_data

## initialization
print('raw data multimodal network')

net=complete_data(num_classes=2)

alldata=fmri(transform=ChannelSelect())

ind=list(range(len(alldata)))
random.shuffle(ind)

# split training set and testing set 
train_ind=ind[:800]
trainset=torch.utils.data.Subset(alldata,train_ind)
test_ind=ind[800:]
testset=torch.utils.data.Subset(alldata,test_ind)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=8,
                shuffle=False, num_workers=4)
testloader=torch.utils.data.DataLoader(testset, batch_size=8,
                shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# hyperparameters
parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch', default=20, type=int,
                    help='epoches for training')
parser.add_argument('--decay', default=0.0, type=float,
                    help='weight decay l2 regularization')

args = parser.parse_args()

optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=args.decay)
criterion = nn.CrossEntropyLoss()
epochs=args.epoch

train_acc = []
test_acc = []

for e in range(epochs):
    print('epoch: '+str(e+1))
    net.train()
    correct=0
    total=0

    ### train

    for i, data in enumerate(trainloader):
        print('  minibatch: '+str(i+1))
        img=(data[0]).to(device)
        tseries=(data[1]).type(torch.FloatTensor)
        tseries=tseries.to(device)
        labels=(data[2]).to(device)

        optimizer.zero_grad()
        outputs=net(img, tseries)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        accuracy = 100*correct/total
        train_acc.append(accuracy)

    
    print('epoch[{}]\t loss: {} \ttrain acc: {}%'.format(e+1, loss, accuracy))
    
    ## evaluate
    net.eval()

    accuracy=0
    correct_test=0
    total_test=0
    
    for i, data_test in enumerate(testloader):
        img=(data_test[0]).to(device)
        tseries=(data_test[1]).type(torch.FloatTensor)
        tseries=tseries.to(device)
        labels=(data_test[2]).to(device)

        outputs_test=net(img, tseries)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

        accuracy=100*correct_test/total_test
        test_acc.append(accuracy)
    
    print('testing acc: {}%'.format(max(test_acc)))

