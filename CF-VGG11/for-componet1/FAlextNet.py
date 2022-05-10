from torch import nn
from torchvision import models
import torch.nn.functional as F

class FAlexNet(nn.Module):
    def __init__(self):
        super(FAlexNet,self).__init__() #基类初始化first
        self.name = 'FAlexNet'
        # alextnet要求图像输入为
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1) # 输出为 1(batch_size) * 1 * 49 * 49

        # self.bn1=nn.BatchNorm2d(96) # 经过一次批归一化，将数据拉回到正态分布

        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2) # 输出为

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1)

        # self.bn2=nn.BatchNorm2d(256)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.maxpooling3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear1 = nn.Linear(5 *5 *256, 2048) # 输入通道, 输出通道

        self.dropout = nn.Dropout(0.5)

        self.linear2 = nn.Linear(2048, 1024) # 输入通道, 输出通道
        # 以下是对AlexNet的更改部分
        self.dropout2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(1024, 3600) # 输入通道, 输出通道

    # 会被自动调用, module(xxx)  -->  module.forward(xxx)
    def forward(self, x):
        # batch_size

        out=self.conv1(x)   # 输出 batch_size * 96(通道数) * 58 * 58
        # out=self.bn1(out)   # batch_norm似乎在我的数据集中没有必要
        out=F.relu(out)         # 输出 batch_size * 96(通道数) * 58 * 58
        out=self.maxpooling1(out)# 输出 batch_size * 96(通道数) * (58-3)/2 + 1 * 28

        out=self.conv2(out) # 输出 batch_size * 256(通道数) * (28-3)/1 + 1 * 26
        # out=self.bn2(out)
        out=F.relu(out) # 输出 batch_size * 256(通道数) * (24-3)/1 + 1 * 26
        out=self.maxpooling2(out) # 输出 batch_size * 256(通道数) * (26-3)/2 + 1 * 12
        
        out=self.conv3(out) # 输出 batch_size * 384(通道数) * (12-3+2*1)/1 + 1 * 12
        out=F.relu(out)  # 输出 batch_size * 384(通道数) * (12-3+2*1)/1 + 1 * 12
        out = self.conv4(out)   # 输出 batch_size * 384(通道数) * (12-3+2*1)/1 + 1 * 12
        out=F.relu(out) # 输出 batch_size * 384(通道数) * (12-3+2*1)/1 + 1 * 12
        out = self.conv5(out)   # 输出 batch_size * 384(通道数) * (12-3+2*1)/1 + 1 * 12
        out=F.relu(out)    # 输出 batch_size * 256(通道数) * (12-3+2*1)/1 + 1 * 12

        out=self.maxpooling3(out)   # 输出 batch_size * 256(通道数) * (12-3)/2 + 1 * 5

        out=out.view(-1,5*5*256) # 256*5*5 # 此处填写256*5*5 填写-1出来大小是一样的

        out = self.linear1(out)

        out=F.relu(out)

        out=self.dropout(out) # dropout 增强数据, 减少过拟合的可能性

        out=self.linear2(out)

        out=F.relu(out)

        out=self.dropout2(out)

        out=self.linear3(out)

        return out