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
    print(Net.cal_IoU(17, 19, 13, 5))
