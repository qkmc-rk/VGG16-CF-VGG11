from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__() #基类初始化first
        self.name = 'SimpleCNN'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64 , kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(64 * 13 * 13, 2048) # 输入通道, 输出通道
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 1024) # 输入通道, 输出通道
        self.linear3 = nn.Linear(1024, 3600) # 输入通道, 输出通道
        
    # 会被自动调用, module(xxx)  -->  module.forward(xxx)
    def forward(self, x):
        input_size = x.size(0) # batch_size  224    60
        x = self.conv1(x)   # 输出: batch_size * 32 * 58 * 58
        x = self.relu1(x)   # 输出: batch_size * 32 * 58 * 58

        x = self.maxpooling1(x) # 输出: batch_size * 32 * 29 * 29

        x = self.conv2(x)   # 输出: batch_size * 64 * 27 * 27
        x = self.relu2(x)   # 输出: batch_size * 64 * 27 * 27

        x = self.maxpooling2(x) # 输出: batch_size * 64 * 13 * 13

        x = x.view(input_size, -1)  # 4维张量转换为2维   batch_size * 64 * 13 * 13 , 这个-1表示64 * 13 * 13
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x