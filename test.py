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
import torchvision.models
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]


if __name__ == '__main__':
    a = torch.FloatTensor(range(0, 10))
    print(a)
    a = a.reshape((5, -1))
    print(a)
