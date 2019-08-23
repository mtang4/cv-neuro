import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import random
import os
import pickle

import argparse 
import numpy as np
import time

from loadfmri import fmridata
from corrdata import corrdata
from time import gmtime, strftime
# from tensorboardX import SummaryWriter

# store_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
# tf_writer=SummaryWriter(log_dir=os.path.join('logs/fmrixroi',store_name))

from new3D import new_conv
from simple3D import simple_conv
from corrinputs import corrmat

parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch', default=30, type=int,
                    help='epoches for training')
parser.add_argument('--decay', default=0.0, type=float,
                    help='weight decay l2 regularization')

args = parser.parse_args()

### initialize model + data
print('fmrixroi correlation')
net=new_conv(num_classes=2)

### initialize dataset
'''
train_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_train_list.txt'
test_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_val_list.txt'
traindata = corrdata(train_img_file, load_fmri=True)
testdata = corrdata(test_img_file, load_fmri=True)

trainloader=torch.utils.data.DataLoader(traindata, batch_size=8,
                shuffle=True,num_workers=4)
testloader=torch.utils.data.DataLoader(testdata, batch_size=8,
                shuffle=False,num_workers=4)
'''

path='/vulcan/scratch/mtang/code/neuroimaging/network_construct/'
img_file=path+'all_subjects.txt'
alldata=corrdata(img_file, load_fmri=True)
ind=list(range(len(alldata)))
# random.shuffle(ind)

train_ind=ind[:800]
trainset=torch.utils.data.Subset(alldata,train_ind)

test_ind=ind[800:]
testset=torch.utils.data.Subset(alldata,test_ind)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=8,shuffle=True,num_workers=4)
torch.save(trainloader, path+'trainset.pt')

testloader=torch.utils.data.DataLoader(testset, batch_size=8,shuffle=True,num_workers=4)
torch.save(testloader, path+'testset.pt')


### training
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
    momentum=0.9, weight_decay=args.decay)
criterion = nn.CrossEntropyLoss()
epochs=args.epoch

train_acc = []
test_acc = []


if torch.cuda.is_available():
    net = net.cuda()

for epoch in range(epochs):
    net.train()
    correct=0
    total=0
    

    for i, data in enumerate(trainloader):
        inputs, labels = data[1].cuda(), data[2].cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        # tf_writer.add_scalar('train/loss', loss, epoch*len(trainloader) + i)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        accuracy = correct/total

    print('epoch[{}]\tloss: {} \ttrain acc: {}'.format(epoch+1, loss, accuracy))
    
    # tf_writer.add_scalar('train/acc', accuracy, epoch)

    correct_test=0
    total_test=0

    ### evaluate
    for i, data_test in enumerate(testloader):
        inputs_test, labels_test = data_test[1].cuda(), data_test[2].cuda()
        outputs_test=net(inputs_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        total_test += labels_test.size(0)
        correct_test += (predicted_test == labels_test).sum().item()
    
    print('testing accuracy: ' + str(100*correct_test/total_test)+'%')
    # tf_writer.add_scalar('test/acc', (100*correct_test/total_test), epoch)

torch.save(net, '/vulcan/scratch/mtang/code/neuroimaging/network_construct/'+'fmrixroi3.pt')