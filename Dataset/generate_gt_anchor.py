# coding=utf-8
import math
import other
import copy
import numpy as np


def generate_gt_anchor(img, box, anchor_width=16, threshold=7):
    """
    把文本框分成一个个16px的小框的
    :param img: input image 输入的图片
    :param box: ground truth box (4 point) 一个文本框的四个点
    :param anchor_width: anchor的长度
    :param threshold:
    :return: tuple (position, h, cy)
    """
    # 看看文本框是不是用float表示的
    if not isinstance(box[0], float):
        box = [float(box[i]) for i in range(len(box))]
    result = []
    # 算一下最左边和最右边在x轴方向都是第几个anchor
    left_anchor_num = int(math.floor(min(box[0], box[6]) / anchor_width))
    right_anchor_num = int(math.floor(max(box[2], box[4]) / anchor_width))

    # 规定了一个距离gt box边界的最小值
    if left_anchor_num * anchor_width + threshold < min(box[0], box[6]):
        left_anchor_num = left_anchor_num + 1

    if right_anchor_num * anchor_width + threshold > max(box[2], box[4]):
        right_anchor_num = right_anchor_num - 1

    # 超出图像边界的就不要了
    if right_anchor_num * 16 + 15 > img.shape[1]:
        right_anchor_num -= 1
    # 产生每个anchor最左边像素和最右边像素的横坐标
    position_pair = [(i * anchor_width, (i + 1) * anchor_width - 1)
                     for i in range(left_anchor_num, right_anchor_num + 1)]
    # 针对每个anchor算一下上下的y坐标
    y_top, y_bottom = cal_y_top_and_bottom(img, position_pair, box)
    if len(y_top) != len(position_pair):
        return result
    # 返回结果，position是左到右第几个anchor（从0开始）
    for i in range(len(position_pair)):
        position = int(position_pair[i][0] / anchor_width)
        # h是anchor框的高度
        h = y_bottom[i] - y_top[i] + 1
        # cy就是论文里的cy（每个anchor中心点y坐标）
        cy = (float(y_bottom[i]) + float(y_top[i])) / 2.0
        # if i == 0 or i == len(position_pair) - 1:
        #     result.append((position, cy, h, 1))
        # else:
        #     result.append((position, cy, h, 0))
        result.append((position, cy, h, i))
    return result


def cal_y_top_and_bottom(raw_img, position_pair, box):
    """
    根据每个anchor的细条，算它的高度的，就是上下y坐标
    用了巨蠢的方法，就是先把图像的一个通道全变成0
    然后利用B=255的颜色画groundtruth文本框
    最后每个anchor的细条里找y坐标最大的和最小的像素，这个就是上下y坐标
    :param raw_img: 原始图像
    :param position_pair: for example:[(0, 15), (16, 31), ...]
    :param box: gt box (4 point)
    :return: top and bottom coordinates for y-axis
    """
    img = copy.deepcopy(raw_img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j, 0] = 0
    img = other.draw_box_4pt(img, box, color=(255, 0, 0))
    # # y_top, y_bottom = f.find_top_bottom(img, position_pair)
    y_top = []
    y_bottom = []
    # height = img.shape[0]
    # top_flag = False
    # bottom_flag = False
    # for k in range(len(position_pair)):
    #     for y in range(0, height):
    #         for x in range(position_pair[k][0], position_pair[k][1] + 1):
    #             if img[y, x, 0] == 255:
    #                 y_top.append(y)
    #                 top_flag = True
    #                 break
    #         if top_flag is True:
    #             break
    #     for y in range(height - 1, -1, -1):
    #         for x in range(position_pair[k][0], position_pair[k][1] + 1):
    #             if img[y, x, 0] == 255:
    #                 y_bottom.append(y)
    #                 bottom_flag = True
    #                 break
    #         if bottom_flag is True:
    #             break
    #     top_flag = False
    #     bottom_flag = False
    for k in range(len(position_pair)):
        y_coord = np.where(img[:, position_pair[k][0]:(position_pair[k][1] + 1), 0] == 255)[0]
        if len(y_coord) == 0:
            continue
        y_top.append(np.min(y_coord))
        y_bottom.append(np.max(y_coord))
    return y_top, y_bottom
