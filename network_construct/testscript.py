from corrnet_mcb import classifier
import torch

net=classifier(num_classes=2)
net=net.cuda()
data=torch.load('trainset.pt')
correct=0

for i, (roi, img, label) in enumerate(data):
    roi=roi.cuda()
    img=img.cuda()
    label=label.cuda()
    outputs=net(roi,img)

    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == label).sum().item()
    print('correct: '+str(correct))
    break