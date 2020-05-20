## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #image size = 224x224
        #architecture: conv1,maxpool,con2,maxpool,fc1,fc1_drop,fc2
        #define sizes and number of nodes in each layer
        conv1size=4
        conv1num=32
        conv2size=3
        conv2num=64
        conv3size=2
        conv3num=128
        conv4size=1
        conv4num=256
        fc1size=4000
        dropp=0.2
        fc2size=2000
        #calculate output size after conv1 and pool1
        OutputSizeConv1=((224-conv1size)/1+1)//2
        #calculate output size after conv2 and pool1
        OutputSizeConv2=((OutputSizeConv1-conv2size)/1+1)//2
        #calculate output size after conv3 and pool1
        OutputSizeConv3=((OutputSizeConv2-conv3size)/1+1)//2
        #calculate output size after conv4 and pool1
        OutputSizeConv4=int(((OutputSizeConv3-conv4size)/1+1)//2)
        self.conv1 = nn.Conv2d(1, conv1num, conv1size)
        #maxpool layer
        self.pool = nn.MaxPool2d(2,2)
        #2nd conv layer: 
        self.conv2 = nn.Conv2d(conv1num,conv2num,conv2size)
        #3rd conv layer: 
        self.conv3 = nn.Conv2d(conv2num,conv3num,conv3size)
        #4th conv layer: 
        self.conv4 = nn.Conv2d(conv3num,conv4num,conv4size)
        #dense layer
        self.fc1=nn.Linear(conv4num*OutputSizeConv4*OutputSizeConv4,fc1size)
        #dropout
        self.drop_layer = nn.Dropout(p=dropp)
        #final output
        self.fc2 = nn.Linear(fc1size,fc2size)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        #conv1+pool
        x = self.pool(F.relu(self.conv1(x)))
        #conv2+pool
        x = self.pool(F.relu(self.conv2(x)))
        #conv3+pool
        x = self.pool(F.relu(self.conv3(x)))
        #conv4+pool
        x = self.pool(F.relu(self.conv4(x)))
        #prep for linear layer, flatten
        #print('x size:',x.size())
        x = x.view(x.size(0), -1)
        #dense layer with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
