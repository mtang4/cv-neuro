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

from hao_data import corrdata
from hao2_corrnet import small_class, large_class

from tensorboardX import SummaryWriter

LOAD_FMRI=True
store_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
tf_writer=SummaryWriter(log_dir=os.path.join('logs',store_name))

## initialization
print('correlation multimodal network')

net=large_class(num_classes=2)

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

    ### train
    start_time = time.time()
    # data_bar = tqdm(trainloader)
    # fmri_data = torch.tensor(0.)
    # for i, (fc_data, labels)  in enumerate(data_bar):
    for i, (fc_data, fmri_data, labels)  in enumerate(trainloader):
        data_time = time.time() - start_time
        if torch.cuda.is_available():
            fc_data = fc_data.cuda()
            fmri_data = fmri_data.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs=net(fc_data, fmri_data)

        loss = criterion(outputs, labels)
        tf_writer.add_scalar('train/loss', loss, e*len(trainloader) + i)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_acc = correct / total
        total_time = time.time() - start_time
        # data_bar.set_postfix(Time='%.3f' %(total_time),
        #     Data_time='%.3f' %(data_time),
        #     loss='%.4f' %(loss),
        #     train_acc='%.4f' %(train_acc))
        print('Epoch {}\t [{}]/[{}]\t Time: {}\t Data_time: {}\t Loss: {}\t acc: {}'.format(
            e, i, len(trainloader), round(total_time, 3), round(data_time, 3), round(loss.item(),3), round(train_acc,3)))

    accuracy = 100*correct/total    
    tf_writer.add_scalar('train/acc', accuracy, e)

    ## evaluate
    
    accuracy=0
    correct_test=0
    total_test=0
    
    net.eval()
    # fmri_data = torch.tensor(0.)
    # for i, (fc_data, labels)  in enumerate(testloader):
    for i, (fc_data, fmri_data, labels)  in enumerate(testloader):
        if torch.cuda.is_available():
            fc_data = fc_data.cuda()
            fmri_data = fmri_data.cuda()
            labels = labels.cuda()

        outputs_test=net(fc_data, fmri_data)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

        # accuracy=100*correct_test/total_test
        # test_acc.append(accuracy)
    test_acc = 100*correct_test/total_test
    # print('testing acc: {}%'.format(max(test_acc)))
    tqdm.write('testing acc: {}%'.format(test_acc))
    tf_writer.add_scalar('test/acc', test_acc, e)


