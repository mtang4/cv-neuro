import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import os
import random

from corrdata import corrdata
from hao_corrnet import complete_corr, fmrixroi

## initialize
net=complete_corr(num_classes=2)
print('net initialized.')

alldata=corrdata()
ind=list(range(len(alldata)))
random.shuffle(ind)

# split training set and testing set 
train_ind=ind[:800]
trainset=torch.utils.data.Subset(alldata,train_ind)
test_ind=ind[800:]
testset=torch.utils.data.Subset(alldata,test_ind)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=False,num_workers=0)
testloader=torch.utils.data.DataLoader(testset, batch_size=16,shuffle=False,num_workers=0)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)
if torch.cuda.is_available():
    net = net.cuda()

print('correlation multimodal network')

optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
epochs=20

train_acc = []
test_acc = []

for e in range(epochs):
    print('epoch: '+str(e+1))
    net.train()
    correct=0
    total=0

    ### train
    # import pdb;pdb.set_trace()
    for i, data in enumerate(trainloader):
        print('  minibatch: '+str(i+1))
        img = data[0].cuda()
        tseries = data[1].cuda()
        labels = data[2].cuda()
        # img=(data[0]).to(device)
        # tseries=(data[1]).to(device)
        # labels=(data[2]).to(device)

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
    
    accuracy=0
    correct_test=0
    total_test=0
    
    net.eval()

    for i, data_test in enumerate(testloader):
        # img=(data_test[0]).to(device)
        # tseries=(data_test[1]).to(device)
        # labels=(data_test[2]).to(device)
        img = data[0].cuda()
        tseries = data[1].cuda()
        labels = data[2].cuda()

        outputs_test=net(img, tseries)
        _, predicted_test = torch.max(outputs_test.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

        accuracy=100*correct_test/total_test
        test_acc.append(accuracy)
    
    print('testing accuracy: ' + str(accuracy)+'%')
    print('max test acc: {}'.format(max(test_acc)))

