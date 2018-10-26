import torch.nn as nn
import torch.nn.functional as F


class VGG_16(nn.Module):
    """
    VGG-16 without pooling layer before fc layer
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        self.convolution1_1 = nn.Conv2d(3, 3, 64, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 3, 64, padding=1)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.convolution2_1 = nn.Conv2d(64, 3, 128, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 3, 128, padding=1)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.convolution3_1 = nn.Conv2d(128, 3, 256, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 3, 256, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 3, 256, padding=1)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.convolution4_1 = nn.Conv2d(256, 3, 512, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 3, 512, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 3, 512, padding=1)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.convolution5_1 = nn.Conv2d(512, 3, 512, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 3, 512, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 3, 512, padding=1)

    def forward(self, x):
        x = F.relu(self.convolution1_1(x), inplace=True)
        x = F.relu(self.convolution1_2(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2_1(x), inplace=True)
        x = F.relu(self.convolution2_2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3_1(x), inplace=True)
        x = F.relu(self.convolution3_2(x), inplace=True)
        x = F.relu(self.convolution3_3(x), inplace=True)
        x = self.pooling3(x)
        x = F.relu(self.convolution4_1(x), inplace=True)
        x = F.relu(self.convolution4_2(x), inplace=True)
        x = F.relu(self.convolution4_3(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution5_1(x), inplace=True)
        x = F.relu(self.convolution5_2(x), inplace=True)
        x = F.relu(self.convolution5_3(x), inplace=True)
        return x


class BLSTM(nn.Module):
    def __init__(self, kernel_size, channel, hidden_unit, padding=None, stride=1, bidirectional=True):
        """
        :param kernel_size: im2col kernel size
        :param channel: lstm input channel num
        :param hidden_unit: lstm hidden unit
        :param padding: im2col padding
        :param stride: im2col stride
        :param bidirectional:
        """
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(kernel_size * kernel_size * channel, channel, bidirectional=bidirectional)

    def forward(self, x):
        pass