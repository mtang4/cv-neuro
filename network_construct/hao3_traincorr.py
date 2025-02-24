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
from hao3_corrnet import classifier

from tensorboardX import SummaryWriter

LOAD_FMRI=True
store_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
tf_writer=SummaryWriter(log_dir=os.path.join('logs',store_name))

## initialization
print('correlation multimodal network')

net=classifier(num_classes=2)

train_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_train_list.txt'
val_img_file = '/vulcan/scratch/mtang/code/neuroimaging/network_construct/abide_val_list.txt'
traindata = corrdata(train_img_file, load_fmri=LOAD_FMRI)
testdata = corrdata(val_img_file, load_fmri=LOAD_FMRI)

trainloader=torch.utils.data.DataLoader(traindata, batch_size=8,
                shuffle=True,num_workers=4)
testloader=torch.utils.data.DataLoader(testdata, batch_size=8,
                shuffle=False,num_workers=4)


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

for e in range(epochs):
    print('epoch: '+str(e))
    net.train()
    correct=0
    total=0

    ### train
    # start_time = time.time()
    for i, (fc_data, fmri_data, labels)  in enumerate(trainloader):
        # data_time = time.time() - start_time
        if torch.cuda.is_available():
            fc_data = fc_data.cuda()
            fmri_data = fmri_data.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs, r_out, f_out = net(fc_data, fmri_data)

        loss_0 = criterion(outputs, labels)
        loss_1 = criterion(r_out, labels)
        loss_2 = criterion(f_out, labels)
        loss = loss_0 + loss_1 + loss_2
        tf_writer.add_scalar('train/loss', loss, e*len(trainloader) + i)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_acc = correct/total
        # total_time = time.time() - start_time

        # print('Epoch {}\t [{}]/[{}]\t Time: {}\t Data_time: {}\t Loss: {}\t acc: {}'.format(
        #    e, i, len(trainloader), round(total_time, 3), round(data_time, 3), round(loss.item(),3), round(train_acc,3)))

        print('Epoch {}\t [{}]/[{}]\t Loss: {}\t acc: {}'.format(
           e, i, len(trainloader), round(loss.item(),3), round(train_acc,3)))

    accuracy = correct/total    
    tf_writer.add_scalar('train/acc', accuracy, e)

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

        outputs_test, r_test, f_test =net(fc_data, fmri_data)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

    test_acc = 100*correct_test/total_test
    tqdm.write('testing acc: {}%'.format(test_acc))
    tf_writer.add_scalar('test/acc', test_acc/100, e)


