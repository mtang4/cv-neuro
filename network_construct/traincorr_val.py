### correlation multimodal model, with k-fold cross val

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import os
import random
import argparse
import time
from time import gmtime, strftime
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from corrdata import corrdata
from corrnet2 import classifier


from tensorboardX import SummaryWriter
from sklearn.model_selection import RepeatedKFold, KFold

store_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
tf_writer=SummaryWriter(log_dir=os.path.join('logs',store_name))

## initialization
print('correlation multimodal network')

net=classifier(num_classes=2)


img_file='/vulcan/scratch/mtang/code/neuroimaging/network_construct/all_subjects.txt'
alldata=corrdata(img_file, load_fmri=True)


if torch.cuda.is_available():
    net = net.cuda()

# hyperparameters
parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch', default=20, type=int,
                    help='epoches for training')
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay l2 regularization')

args = parser.parse_args()
print('args: {}'.format(args))
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
                weight_decay=args.decay)

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()
epochs=args.epoch
# import pdb;pdb.set_trace()

train_acc = []
test_acc = []
losscount=0
# kf = KFold(n_splits=5, shuffle=True, random_state=None)
rkf = RepeatedKFold(n_splits=5, n_repeats=args.epoch, random_state=None)

# for e in range(epochs):
    # print('epoch: '+str(e+1))
net.train()
correct=0
total=0
k=0
c=0
e=0

for train, test in rkf.split(alldata):
    # print('  k-fold: '+str(k+1))
    train_index=train.tolist()
    test_index=test.tolist()
        
    trainset=torch.utils.data.Subset(alldata, train_index)
    testset=torch.utils.data.Subset(alldata, test_index)
        
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=8,
            shuffle=True,num_workers=4)
    testloader=torch.utils.data.DataLoader(testset, batch_size=8,
            shuffle=False,num_workers=4)

    ### train
    for i, (fc_data, fmri_data, labels)  in enumerate(trainloader):
        if torch.cuda.is_available():
            fc_data = fc_data.cuda()
            fmri_data = fmri_data.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs=net(fc_data, fmri_data)

        loss = criterion(outputs, labels)
        tf_writer.add_scalar('train/loss', loss, losscount)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_accuracy = correct/total

        print('Epoch {}\t [{}]/[{}]\tk-Fold: {}\t Loss: {}\t Acc: {}'.format(
            e+1, i+1, len(trainloader), k+1, round(loss.item(),3), round(train_accuracy,3)))
        
        losscount+=1

    tf_writer.add_scalar('train/acc', train_accuracy, (5*e)+k)
    train_acc.append(train_accuracy)

    ## evaluate
        
    accuracy=0
    correct_test=0
    total_test=0
        
    net.eval()
    for i, (fc_data, fmri_data, labels)  in enumerate(testloader):
        if torch.cuda.is_available():
            fc_data = fc_data.cuda()
            fmri_data = fmri_data.cuda()
            labels = labels.cuda()

        outputs_test=net(fc_data, fmri_data)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

    test_accuracy = correct_test/total_test
    tqdm.write('testing acc: {}%'.format(100*test_accuracy))
    tf_writer.add_scalar('test/acc', test_accuracy, (5*e)+k)
    test_acc.append(test_accuracy)

    k+=1
    if k%5==0:
        e+=1
        k=0