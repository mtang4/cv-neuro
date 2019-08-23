import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tdata
import random
import os
import pickle

from loadinput import inputs
from simple3D import simple_conv


import numpy as np
from tensors import tensors
from generated import generated
from sklearn.model_selection import RepeatedKFold, KFold

net=simple_conv(num_classes=2)
net=net.float()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('initializing datasets')
batch_size=16
### initialize data
allData=inputs()


kf = KFold(n_splits=5, shuffle=True, random_state=None)

# trainSet, testSet=generated(16)

### check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

print('begin training:')
### training
for epoch in range(2):  # loop over the dataset multiple times
      print('  epoch: '+str(epoch+1))
      running_loss = 0.0
      running_corrects=0.0
      batch_size=16

      for train_index, test_index in kf.split(allData):
            train_index=train_index.tolist()
            test_index=test_index.tolist()

            trainSet=[allData[i] for i in train_index]
            testSet=[allData[j] for j in test_index]

            # training
            for i in range(len(trainSet)):
                  print('    mini-batch: '+str(i))
                  # inputs, labels = data             # cpu data
                  data=trainSet[i]
                  inputs, labels = (data[0]).to(device), data[1].to(device)
                  optimizer.zero_grad()
                  
                  # forward + backward + optimize
                  outputs = net(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  
                  # running_loss += loss.item()
                  
                  # accuracy
                  _, predict = torch.max(outputs, 1)
                  running_corrects += torch.sum(predict == labels.data)
                  train_acc = running_corrects.double()/batch_size
                  print("      train accuracy: {:.3f}".format(train_acc))
                  batch_size+=16
            
            print('finished training')

            ### testing
            correct = 0
            total = 0
            with torch.no_grad():
                  for i in range(len(testSet)):
                        data=testSet[i]
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            
            print('testing accuracy: ' + str(100 * correct/total)+'%')

      



