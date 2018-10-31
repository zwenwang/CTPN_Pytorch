import torch.nn as nn
import torch.nn.functional as F
import Net
import numpy as np
import random
import torch
from torch.autograd import Variable


class VGG_16(nn.Module):
    """
    VGG-16 without pooling layer before fc layer
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        self.convolution1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.convolution2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.convolution3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.convolution4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.convolution5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 512, 3, padding=1)

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
    def __init__(self, channel, hidden_unit, weight_init=False, bidirectional=True):
        """
        :param channel: lstm input channel num
        :param hidden_unit: lstm hidden unit
        :param bidirectional:
        :param weight_init: use Gaussian distribution to init weight or not
        """
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)
        if weight_init:
            for i in range(len(self.lstm.all_weights)):
                for j in range(len(self.lstm.all_weights[0])):
                    torch.nn.init.normal_(self.lstm.all_weights[i][j], std=0.01)

    def forward(self, x):
        """
        WARNING: The batch size of x must be 1.
        """
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent


class CTPN(nn.Module):
    def __init__(self, weight_init=False):
        """
        :param weight_init: use Gaussian distribution to init weight or not
        """
        super(CTPN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('VGG_16', VGG_16())
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', Net.Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * 512, 128, weight_init=True))
        self.FC = nn.Conv2d(256, 512, 1)
        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)
        self.score = nn.Conv2d(512, 2 * 10, 1)
        self.side_refinement = nn.Conv2d(512, 10, 1)
        if weight_init:
            torch.nn.init.normal_(self.FC.weight, mean=0, std=0.01)
            torch.nn.init.constant_(self.FC.bias, val=0)

            torch.nn.init.normal_(self.vertical_coordinate.weight, mean=0, std=0.01)
            torch.nn.init.constant_(self.vertical_coordinate.bias, val=0)

            torch.nn.init.normal_(self.score.weight, mean=0, std=0.01)
            torch.nn.init.constant_(self.score.bias, val=0)

            torch.nn.init.normal_(self.side_refinement.weight, mean=0, std=0.01)
            torch.nn.init.constant_(self.side_refinement.bias, val=0)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        side_refinement = self.side_refinement(x)
        return vertical_pred, score, side_refinement


class CTPN_Loss(nn.Module):
    def __init__(self):
        super(CTPN_Loss, self).__init__()
        self.Ns = 128
        self.ratio = 0.5
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.Ls_cls = nn.CrossEntropyLoss()
        self.Lv_reg = nn.SmoothL1Loss()
        self.Lo_reg = nn.SmoothL1Loss()

    def forward(self, score, vertical_pred, side_refinement, positive, negative, vertical_reg):
        """
        :param score: prediction score
        :param vertical_pred: prediction vertical coordinate
        :param side_refinement: prediction side refinement
        :param positive: ground truth positive fine-scale box
        :param negative: ground truth negative fine-scale box
        :param vertical_reg: ground truth vertical regression
        :return: total loss
        """
        # calculate classification loss
        positive_num = min(int(self.Ns * self.ratio), len(positive))
        negative_num = self.Ns - positive_num
        positive_batch = random.sample(positive, positive_num)
        negative_batch = random.sample(negative, negative_num)
        cls_loss = 0
        for p in positive_batch:
            cls_loss += self.Ls_cls(score[0, p[2] * 2: ((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0),
                                    torch.LongTensor([1]).cuda())
        for n in negative_batch:
            cls_loss += self.Ls_cls(score[0, n[2] * 2: ((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0),
                                    torch.LongTensor([0]).cuda())
        cls_loss = cls_loss / self.Ns

        # calculate vertical coordinate regression loss
        v_reg_loss = 0
        Nv = len(vertical_reg)
        for v in vertical_reg:
            v_reg_loss += self.Lv_reg(vertical_pred[0, v[2] * 2: ((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0),
                                      torch.FloatTensor([v[3], v[4]]).unsqueeze(0).cuda())
        v_reg_loss = v_reg_loss / float(Nv)

        loss = cls_loss + v_reg_loss
        return loss, cls_loss, v_reg_loss
