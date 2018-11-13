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
from other.lib import nms
import math
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]


if __name__ == '__main__':
    net = Net.CTPN()
    net.load_state_dict(torch.load('./model/ctpn-9-end.model'))
    print(net)
    net.eval()
    # im = cv2.imread('/home/wzw/ICDAR2015/test_image/img_99.jpg')
    im = cv2.imread('/home/wzw/990714656.jpg')
    # gt = Dataset.port.read_gt_file('/home/wzw/ICDAR2015/test_gt/gt_img_99.txt')
    im = cv2.resize(im, (0, 0), fx=0.3, fy=0.3)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    v, score, side = net(img, val=True)
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > 0.7:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))
    print(v.shape)
    print(score.shape)
    print(side.shape)
    # print(result)
    # for box in result:
    #     im = other.draw_box_h_and_c(im, box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
    for_nms = []
    for box in result:
        pt = other.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
        for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = nms.cpu_nms(for_nms, 0.3)

    for i in nms_result:
        vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
        vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
        cya = for_nms[i, 5] * 16 + 7.5
        ha = anchor_height[int(for_nms[i, 7])]
        cy = vc * ha + cya
        h = math.pow(10, vh) * ha
        # other.draw_box_2pt(im, for_nms[i, 0:4])
        other.draw_box_h_and_c(im, int(for_nms[i, 6]), cy, h)

    # for gt_box in gt:
    #     im = other.draw_box_4pt(im, gt_box, (255, 0, 0))

    cv2.imshow('kk', im)
    cv2.waitKey(0)
