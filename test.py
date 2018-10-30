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
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]


if __name__ == '__main__':
    i = np.array([1, 2, 3, 4])
    l = np.array([1, 0])
    i = i.reshape([2, 2])
    i = torch.FloatTensor(i)
    l = torch.LongTensor(l)
    print(i.shape)
    print(l.shape)
    c = torch.nn.CrossEntropyLoss()
    loss = c(i, l)
    print(loss)
