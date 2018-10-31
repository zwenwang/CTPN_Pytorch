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
    a = torch.IntTensor([1, 2, 3])
    b = torch.IntTensor([1, 2, 3])
    print(a == b)
