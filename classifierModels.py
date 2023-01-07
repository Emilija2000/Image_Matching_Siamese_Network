import numpy as np
import torch
import torch.nn as nn

class FCnet(nn.Module):
    def __init__(self,in_features,out_features=1,pretrained=None):
        super(FCnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,out_features)
        )
        self.sigm = nn.Sigmoid()
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def forward(self, x):
        output = self.fc(x)
        output = self.sigm(output)
        return output
    

class SimpleNetwork(nn.Module):
    def __init__(self,pretrained=None,numclasses=50):
        super(SimpleNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(11*11*8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, numclasses)
        )
        self.softmax = nn.Softmax()
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        #output = self.softmax(output)
        return output

class MobileNet(nn.Module):
    def __init__(self,pretrained=None,numclasses=50):
        super(MobileNet, self).__init__()
        self.cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
        #from https://github.com/chenyaofo/pytorch-cifar-models/tree/master/pytorch_cifar_models
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        #for param in self.cnn.layer1[2].parameters():
        #    param.requires_grad=True
            
        in_params = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Sequential(
            nn.Linear(in_params,64),
            nn.ReLU(),
            nn.Linear(64,numclasses)
        )
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def forward(self, x):
        output = self.cnn(x)
        return output

class Resnet20(nn.Module):
    def __init__(self,pretrained=None,numclasses=50):
        super(Resnet20, self).__init__()
        self.cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
        #from https://github.com/chenyaofo/pytorch-cifar-models/tree/master/pytorch_cifar_models
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        #for param in self.cnn.layer1[2].parameters():
        #    param.requires_grad=True
            
        in_params = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_params,64),
            nn.ReLU(),
            nn.Linear(64,numclasses)
        )
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def forward(self, x):
        output = self.cnn(x)
        return output
    
class EfficientNet(nn.Module):
    def __init__(self,pretrained=None,numclasses=50):
        super(EfficientNet, self).__init__()
        self.cnn = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.classifier.parameters():
            param.requires_grad=True
            
        in_params = self.cnn.classifier.fc.in_features
        self.cnn.fc = nn.Sequential(
            #nn.Linear(in_params,64),
            #nn.ReLU(),
            #nn.Linear(64,numclasses)
            nn.Linear(in_params,numclasses)
        )
        if pretrained!=None:
            self.load_state_dict(torch.load(pretrained))
        
    def forward(self, x):
        output = self.cnn(x)
        return output
   