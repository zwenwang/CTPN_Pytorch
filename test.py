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


if __name__ == '__main__':
    net = Net.CTPN()
    print(net)
    img = cv2.imread('./img_112.jpg')
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]
    img = torch.FloatTensor(img)
    a, b, c = net(img)
    print(a.shape, b.shape, c.shape)
