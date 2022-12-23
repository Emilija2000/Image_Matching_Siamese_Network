import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self,distance_type=None):
        super(Siamese,self).__init__()
        
        if distance_type=='euclidian':
            self.distance = nn.PairwiseDistance(p=2,keepdim=True)
        elif distance_type=='abs':
            self.distance = nn.PairwiseDistance(p=1,keepdim=True)
        elif distance_type=='cosine':
            self.distance = nn.CosineSimilarity(dim=1)
        else:
            self.distance = nn.PairwiseDistance(p=2)
        

class SimpleSiameseNetwork(Siamese):
    def __init__(self,distance_type,pretrained=None):
        super(SimpleSiameseNetwork, self).__init__(distance_type)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*11*11, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        if pretrained!=None:
            self.load_state_dict(pretrained)
        
    def __forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        dist = self.distance(output1,output2)
        return dist
    

class Resnet20Siamese(Siamese):
    def __init__(self,distance,pretrained=None):
        super(Resnet20Siamese, self).__init__(distance_type=distance)
        resnet20 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        #from https://github.com/chenyaofo/pytorch-cifar-models/tree/master/pytorch_cifar_models
        self.cnn = nn.Sequential(*(list(resnet20.children())[:-1]))
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        if pretrained!=None:
            self.load_state_dict(pretrained)
        
    def __forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        dist = self.distance(output1,output2)
        return dist
        
        