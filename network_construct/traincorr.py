### correlation multimodal model w/o cross val

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import os
import random
import argparse
import time
from time import gmtime, strftime
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from corrdata import corrdata
from corrnet import classifier

from tensorboardX import SummaryWriter

store_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
tf_writer=SummaryWriter(log_dir=os.path.join('logs',store_name))

## initialization
print('correlation multimodal network')

net=classifier(num_classes=2)

trainloader=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/trainset.pt')
testloader=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/testset.pt')

if torch.cuda.is_available():
    net = net.cuda()

# hyperparameters
parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('--lr1', default=0.0001, type=float,
                    metavar='LR', help='feature extractor learning rate')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='classifier learning rate')
parser.add_argument('--epoch', default=20, type=int,
                    help='epoches for training')
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay l2 regularization')

args = parser.parse_args()
print('args: {}'.format(args))
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
                weight_decay=args.decay)
'''
optimizer=torch.optim.SGD([
                {'params': net.roi.parameters()},
                {'params': net.fmri.parameters()},
                {'params': net.fc.parameters(), 'lr': args.lr}
            ], lr=args.lr1, momentum=0.9)
'''
# scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()
epochs=args.epoch
# import pdb;pdb.set_trace()
train_acc = []
test_acc = []

for e in range(epochs):
    print('epoch: '+str(e))
    net.train()
    correct=0
    total=0
    accuracy=0

    ### train
    start_time = time.time()
    for i, (roi, img, labels)  in enumerate(trainloader):
        roi = roi.cuda()
        img = img.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs=net(roi, img)

        loss = criterion(outputs, labels)
        tf_writer.add_scalar('train/loss', loss, e*len(trainloader) + i)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = correct/total
        
        print('Epoch {}\t [{}]/[{}]\t Loss: {}\t acc: {}'.format(
            e, i, len(trainloader), round(loss.item(),3), round(accuracy,3)))

    tf_writer.add_scalar('train/acc', accuracy, e)

    ## evaluate
    
    accuracy_test=0
    correct_test=0
    total_test=0
    
    net.eval()
    
    for i, (roi, img, labels)  in enumerate(testloader):
        if torch.cuda.is_available():
            roi = roi.cuda()
            img = img.cuda()
            labels = labels.cuda()

        outputs_test=net(roi, img)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

        
    accuracy_test = 100*correct_test/total_test
    tqdm.write('testing acc: {}%'.format(accuracy_test))
    tf_writer.add_scalar('test/acc', accuracy_test/100, e)


