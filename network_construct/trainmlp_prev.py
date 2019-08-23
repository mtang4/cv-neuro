from fcnet import MLP
from datanet import conv1D
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import random
import os

from maptensor import maps
from corrdata import corrdata
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import argparse 

parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch', default=500, type=int,
                    help='epoches for training')
parser.add_argument('--decay', default=0.0, type=float,
                    help='weight decay l2 regularization')

args = parser.parse_args()

### initialize model + data
print('ROIxROI correlation')
net=MLP(num_classes=2).float()


allset=maps()
model='roixroi.pt'


ind=list(range(len(allset)))
random.shuffle(ind)

train_ind=ind[:800]
trainset=torch.utils.data.Subset(allset,train_ind)

test_ind=ind[800:]
testset=torch.utils.data.Subset(allset,test_ind)

trainloader=torch.utils.data.DataLoader(trainset,
    batch_size=64,shuffle=True,num_workers=4)
testloader=torch.utils.data.DataLoader(testset,
    batch_size=64,shuffle=False,num_workers=4)


'''
train_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_train_list.txt'
test_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_val_list.txt'
traindata = corrdata(train_img_file, load_fmri=False)
testdata = corrdata(test_img_file, load_fmri=False)

trainloader=torch.utils.data.DataLoader(traindata, batch_size=8,
                shuffle=True,num_workers=4)
testloader=torch.utils.data.DataLoader(testdata, batch_size=8,
                shuffle=False,num_workers=4)
'''

# trainloader=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/trainset.pt')
# testloader=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/testset.pt')

optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
    momentum=0.9, weight_decay=args.decay)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    net = net.cuda()

train_acc = []
valid_acc = []
best_test_acc = 0

for epoch in range(args.epoch):
    net.train()
    correct=0
    total=0
    

    ## train
    for i, (images, labels) in enumerate(trainloader):
        # print('begin training: '+str(i+1))
        optimizer.zero_grad()

        images=images.cuda()
        labels=labels.cuda()

        outputs = net(images.float())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        accuracy = 100*correct/total
        train_acc.append(accuracy)

        if i == len(trainloader) - 1:
            print('epoch[{}]\t loss: {} \taccuracy: {}'.format(
                epoch, loss, accuracy))

    if (epoch+1) % 5 ==0:
        correct_test=0
        total_test=0

        ## evaluate
        for i, (images_test, labels_test) in enumerate(testloader):

            images_test=images_test.cuda()
            labels_test=labels_test.cuda()

            outputs_test=net(images_test.float())
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()
        
        test_acc = correct_test/total_test
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        print('testing accuracy: {}'.format(round(test_acc, 3)))
        print('Best testing accuracy: {}'.format(round(best_test_acc, 3)))
        
# torch.save(net, '/vulcan/scratch/mtang/code/neuroimaging/network_construct/'+'roixroi3.pt')