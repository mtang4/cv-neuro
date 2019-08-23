import torch
import random

def generated(batch_size):
    batch_size=16

    c_labels=[0]*batch_size
    a_labels=[1]*batch_size
    c_list=[]
    a_list=[]

    for i in range(int(400/batch_size)):
        c_tensor=2*torch.randn(batch_size,1,61,73,61)
        a_tensor=2*(torch.randn(batch_size,1,61,73,61)+2)
        c_list.append([c_tensor, torch.tensor(c_labels)])
        a_list.append([a_tensor, torch.tensor(a_labels)])

    trainSet=c_list[0:20]
    trainA=a_list[0:20]

    testSet=c_list[20:len(c_list)]
    testA=a_list[20:len(a_list)]

    for i in range(len(trainA)):
        trainSet.append(trainA[i])

    for i in range(len(testA)):
        testSet.append(testA[i])

    random.shuffle(trainSet)
    random.shuffle(testSet)

    return trainSet, testSet

