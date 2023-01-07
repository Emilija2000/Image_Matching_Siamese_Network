import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets,transforms

import utilities as utils

def readDataFolder(root, numclasses=50):
    dataset = datasets.ImageFolder(root)
    
    idx = []
    for i in range(len(dataset)):
        if dataset.targets[i]<numclasses:
            name = dataset.imgs[i][0]
            if name.rsplit(os.sep,1)[-1][0]!='.':
                idx.append(i)
    idx = np.array(idx)
    
    dataset.samples = np.array(dataset.samples)[idx].tolist()
    dataset.targets = np.array(dataset.targets)[idx].tolist()
    dataset.imgs = np.array(dataset.imgs)[idx].tolist()
    dataset.classes = dataset.classes[:numclasses]
    
    return dataset

def dataSplits(dataset, trainP=0.7, valP=0.2, testP=0.1,seed=42):
    # correct proportions to make sure trainP+valP+testP=1
    sum = trainP+valP+testP
    prop = (np.array([trainP/sum,valP/sum,testP/sum])*len(dataset)).astype(int)
    
    #split data with torch.utils.data.Subset
    train_split,val_split,test_split = random_split(dataset,prop,generator=torch.Generator().manual_seed(seed))
    return train_split,val_split,test_split    
        
class SiameseDataset(Dataset):
    def __init__(self,datasplit,transform=None,sameprob=0.5):
        dataset = datasplit.dataset
        indices = datasplit.indices 
        
        self.imgs = np.array(dataset.imgs)[indices]   
        self.classes = np.array(dataset.targets)[indices]
        
        self.transform = transform
        self.sameprob = sameprob
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        index = np.mod(index,len(self.classes))
        datapoint0 = self.imgs[index]
        
        same = np.random.random() < self.sameprob
        # loop until item of a class we need (same or not) is found
        while True:
            index1 = np.random.randint(0,len(self.classes))
            datapoint1 = self.imgs[index1]        
            if same:
                if datapoint0[1]==datapoint1[1]:
                    break
            else:
                if datapoint0[1]!=datapoint1[1]:
                    break
        
        # load images
        img0 = Image.open(datapoint0[0]).convert('RGB') 
        img1 = Image.open(datapoint1[0]).convert('RGB') 
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        else:
            img0 = transforms.ToTensor(img0)
            img1 = transforms.ToTensor(img1)
            
        # label
        same = torch.from_numpy(np.array([1-same]).astype(np.float32))
        
        return img0,img1,same
    
    def getImageFromClass(self,cls):
        indices = np.where(cls==self.classes)
        num = len(indices[0])
        i =  np.random.randint(0,num)
        img = Image.open(self.imgs[indices[0][i]][0]).convert('RGB') 
        return img,num
        
        
    
class TripletDataset(Dataset):
    def __init__(self,datasplit,transform=None):
        dataset = datasplit.dataset
        indices = datasplit.indices 
        
        self.imgs = np.array(dataset.imgs)[indices]   
        self.classes = np.array(dataset.targets)[indices]
        self.transform = transform
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        datapoint0 = self.imgs[index]
        
        flag1=False 
        flag2=False
        # loop until item of a class we need (same or not) is found
        while True:
            index1 = np.random.randint(0,self.__len__())
            datapoint_temp = self.imgs[index1]        
            
            if datapoint0[1]==datapoint_temp[1] and not(flag1):
                datapoint1=datapoint_temp
                flag1=True
            elif datapoint0[1]!=datapoint_temp[1] and not(flag2):
                datapoint2 = datapoint_temp
                flag2=True
            if flag1 and flag2:
                break
        
        # load images
        img0 = Image.open(datapoint0[0]).convert('RGB') 
        img1 = Image.open(datapoint1[0]).convert('RGB') 
        img2 = Image.open(datapoint2[0]).convert('RGB') 
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img0 = transforms.ToTensor(img0)
            img1 = transforms.ToTensor(img1)
            img2 = transforms.ToTensor(img2)
            
        return img0,img1,img2  
    
class MyImagenetDataset(Dataset):
    def __init__(self,datasplit,transform=None):
        dataset = datasplit.dataset
        indices = datasplit.indices 
        
        self.imgs = np.array(dataset.imgs)[indices]   
        self.classes = np.array(dataset.targets)[indices]
        
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        datapoint = self.imgs[index]
        label = torch.tensor(self.classes[index],dtype=torch.long)
        img = Image.open(datapoint[0]).convert('RGB') 
        img = self.transform(img)
        return img, label
    
    def getImageFromClass(self,cls):
        indices = np.where(cls==self.classes)
        num = len(indices[0])
        i =  np.random.randint(0,num)
        img = Image.open(self.imgs[indices[0][i]][0]).convert('RGB') 
        return img
    
class EmbeddingDataset(Dataset):
    def __init__(self,path):
        data,labels = torch.load(path)
        
        self.data = np.array(data)
        self.labels = np.array(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        d = torch.tensor(self.data[index],dtype=torch.float)
        l = torch.tensor(self.labels[index],dtype=torch.float)
        return d,l

if __name__=='__main__':
    config = utils.load_config()
    
    dataset = readDataFolder(config['DATASET']['root'],config['DATASET']['numclasses'])
    train_split,val_split,test_split = dataSplits(dataset,0.7,0.2,0.1,)
  
    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    test_transforms = train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
        ])

    train_dataset = SiameseDataset(train_split,train_transforms,0.5)
    val_dataset = SiameseDataset(val_split,test_transforms,0.5)
    test_dataset = SiameseDataset(test_split,test_transforms,0.5)
    
    # Visualise
    numex=5 #number of examples
    fig,ax = plt.subplots(2,numex,figsize=(8,3))
    for i in range(numex):
        example = test_dataset[i]
        utils.imshow(example[0],ax[0][i])
        utils.imshow(example[1],ax[1][i])
        ax[0][i].set_title(example[2].item())
    plt.show()