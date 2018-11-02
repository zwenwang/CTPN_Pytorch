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
    # net = Net.CTPN()
    # net.load_state_dict(torch.load('./model/ctpn-epoch9'))
    # print(net)
    # net.eval()
    # im = cv2.imread('/home/wzw/ICDAR2015/test_image/img_2.jpg')
    # img = copy.deepcopy(im)
    # img = img.transpose(2, 0, 1)
    # img = img[np.newaxis, :, :, :]
    # img = torch.Tensor(img)
    # v, score, side = net(img, val=True)
    # result = []
    # for i in range(score.shape[0]):
    #     for j in range(score.shape[1]):
    #         for k in range(score.shape[2]):
    #             if score[i, j, k, 1] > 0.7:
    #                 result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))
    # # print(result)
    # for box in result:
    #     im = other.draw_box_h_and_c(im, box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
    # gt = Dataset.port.read_gt_file('/home/wzw/ICDAR2015/test_gt/gt_img_2.txt')
    # for gt_box in gt:
    #     im = other.draw_box_4pt(im, gt_box, (255, 0, 0))
    #
    # cv2.imshow('kk', im)
    # cv2.waitKey(0)
    pass
