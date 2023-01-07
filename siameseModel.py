import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifierModels import SimpleNetwork,Resnet20,MobileNet

class Siamese(nn.Module):
    def __init__(self,distance_type=None):
        super(Siamese,self).__init__()
        
        if distance_type=='euclidian':
            self.distance = nn.PairwiseDistance(p=2,keepdim=False)
        elif distance_type=='abs':
            self.distance = nn.PairwiseDistance(p=1,keepdim=False)
        elif distance_type=='cosine':
            self.distance = nn.CosineSimilarity(dim=1)
        else:
            self.distance = nn.PairwiseDistance(p=2)
    def forward(self,x1,x2):
        return self.distance(x1,x2)
    

class SimpleSiameseNetwork(Siamese):
    def __init__(self,distance_type,pretrained=None,embsize=64):
        super(SimpleSiameseNetwork, self).__init__(distance_type)
        
        self.network = SimpleNetwork(numclasses=embsize)
        
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def __forward_once(self, x):
        output = self.network(x)
        return output

    def forward(self, input1, input2,output_distance=True):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        dist = self.distance(output1,output2)
        return dist
    
    def getEmb(self,input1,input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        output = torch.concat((output1,output2),dim=1)
        return output.detach().cpu()

class Resnet20Siamese(Siamese):
    def __init__(self,distance,pretrained=None,embsize=64):
        super(Resnet20Siamese, self).__init__(distance_type=distance)
        
        self.network = Resnet20(numclasses=embsize)
        
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def __forward_once(self, x):
        output = self.network(x)
        return output

    def forward(self, input1, input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        dist = self.distance(output1,output2)
        return dist
    
    def getEmb(self,input1,input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        output = torch.concat((output1,output2),dim=1)
        return output.detach().cpu()
    
class Resnet20Binary(nn.Module):
    def __init__(self,pretrained=None,embsize=64):
        super(Resnet20Binary, self).__init__()
        
        self.network = Resnet20(numclasses=embsize)
        
        self.classifier = nn.Sequential(
            nn.Linear(2*embsize,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.sigmoid = nn.Sigmoid()
        
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def __forward_once(self, x):
        output = self.network(x)
        return output

    def forward(self, input1, input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        conc = torch.concat((output1,output2),dim=1)
        out = self.classifier(conc)
        out = self.sigmoid(out)
        return out
    
    def getEmb(self,input1,input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        output = torch.concat((output1,output2),dim=1)
        return output.detach().cpu()
    
class MobileNetSiamese(Siamese):
    def __init__(self,distance,pretrained=None,embsize=64):
        super(MobileNetSiamese, self).__init__(distance_type=distance)
        
        self.network = MobileNet(numclasses=embsize)
        
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def __forward_once(self, x):
        output = self.network(x)
        return output

    def forward(self, input1, input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        dist = self.distance(output1,output2)
        return dist
    
    def getEmb(self,input1,input2):
        output1 = self.__forward_once(input1)
        output2 = self.__forward_once(input2)
        output = torch.concat((output1,output2),dim=1)
        return output.detach().cpu()
    
class MobileNetTriplet(Siamese):
    def __init__(self,distance,pretrained=None,embsize=64):
        super(MobileNetTriplet, self).__init__(distance_type=distance)
        
        self.network = MobileNet(numclasses=embsize)
        
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def __forward_once(self, x):
        output = self.network(x)
        return output

    def forward(self, anchor, positive, negative):
        output1 = self.__forward_once(anchor)
        output2 = self.__forward_once(positive)
        output3 = self.__forward_once(negative)
        
        return output1,output2,output3
    
