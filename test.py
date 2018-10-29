import cv2
import numpy as np
import other
import base64
import os
import copy
import Dataset
import Dataset.port as port
import torch
import Net


# if __name__ == '__main__':
#     # port.create_dataset_icdar2015('/home/wzw/ICDAR2015/train_img', '/home/wzw/ICDAR2015/train_gt',
#     #                               '/home/wzw/lmdb_icdar2015/train')
#     db = Dataset.LmdbDataset('/home/wzw/lmdb_icdar2015/train')
#     print(len(db))
#     im, gt = db[0]
#     print(type(im))
#     im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
#     print(gt)
#     # cv2.imshow('ll', im)
#     # cv2.waitKey(0)
#     gt = [gt.split('|')[i] for i in range(len(gt.split('|')))]
#     tmp = []
#     for b in gt:
#         box = [int(b.split(',')[i]) for i in range(len(b.split(',')))]
#         tmp.append(box)
#     print(tmp)
#     for p in tmp:
#         im = other.draw_box_4pt(im, p)
#     cv2.imshow('asd', im)
#     cv2.waitKey(0)
#
# img_path = './img_112.jpg'
# img = cv2.imread(img_path)
# g = 0
# gt = port.read_gt_file('./gt_img_112.txt', have_BOM=True)
# img = other.draw_box_4pt(img, gt[g], color=(255, 0, 0))
# result = Dataset.generate_gt_anchor(img, gt[g])
# for i in range(len(result['position'])):
#     img = other.draw_box_h_and_c(img, result['position'][i], result['h'][i], result['cy'][i])
# cv2.imshow('kk', img)
# cv2.waitKey(0)


class rnn(torch.nn.Module):
    def __init__(self):
        super(rnn, self).__init__()
        self.lstm = torch.nn.LSTM(3 * 3 * 512, 128, bidirectional=True)

    def forward(self, x):
        x = self.lstm(x)
        return x


class Test_im2col(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Test_im2col, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size, stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    x = np.random.randint(0, 100, (1, 3, 5, 5))
    net = Net.Im2col((3, 3), (1, 1), (1, 1))
    print(x)
    x = torch.FloatTensor(x)
    y = net(x)
    print(y)
