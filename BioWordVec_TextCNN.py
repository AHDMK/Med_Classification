#arguments(input channel, output channel, kernel size, strides, padding)
            
            #layer 1x : 
            # height_out=(h_in-F_h)/S+1=(72-x)/1+1=73-x
            # width_out=(w_in-F_w)/S+1=(384-384)/1+1=1
            # no padding given
            # height_out=(70-x)/(70-x)=1 
            # width_out=1/1=1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
import torch.nn.functional as F 

k1 = max_words+1-3
k2 = max_words+1-4
k3 = max_words+1-5
vector_len = 200
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        D = 300
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=weights.key_to_index['pad'])
        #self.embed = nn.Embedding(199808, D)
        #self.embed.weight.data.copy_(embedding_weights)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(3,vector_len), stride=1,padding=0),  # h = 9-3 +1  and w = 1 output : 7x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k1,1), stride=1)) #1x1
      
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(4,vector_len), stride=1,padding=0), #6x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k2,1), stride=1))  #1x1
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(5,vector_len), stride=1,padding=0), #5x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k3,1), stride=1)) #1X1
       
        self.drop_out = nn.Dropout()
        #concat operation
        self.fc1 = nn.Linear(1 * 1 * 100 * 3, 100)
        self.fc2 = nn.Linear(100, nb_classes)
        
        #self.fc3 = nn.Linear(100,3)
      
    def forward(self, x):
        x=x.to(device)
        x = self.embed(x)
        x = torch.unsqueeze(x, 1)
        x=x.to(device_model)
        #print(x.shape)
        x3 = self.layer1(x)
        #print(x3.shape)
        x4 = self.layer2(x)
        x5 = self.layer3(x)
        x3 = x3.reshape(x3.size(0), -1)
        x4 = x4.reshape(x4.size(0), -1)
        x5 = x5.reshape(x5.size(0), -1)
        #print(x3.shape)
        x3 = self.drop_out(x3)
        x4 = self.drop_out(x4)
        x5 = self.drop_out(x5)
        out = torch.cat((x3,x4,x5),1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        #print(out.shape)
        return(out)