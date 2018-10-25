import Dataset.port as port
import Dataset
from PIL import Image
import cv2
import numpy as np
import other
import base64
import os
import copy


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

img_path = './img_112.jpg'
img = cv2.imread(img_path)
gt = port.read_gt_file('./gt_img_112.txt', have_BOM=True)

img = other.draw_box_4pt(img, gt[0], color=(255, 0, 0))
print(gt[0])


# img_scale, gt_scale = Dataset.scale_img(img, gt)

p, h, c = Dataset.generate_gt_anchor(img, gt[0])
print(p, h, c)
img = other.draw_box_h_and_c(img, p, h, c)
# cv2.imshow('kk', img)
# cv2.waitKey(0)
