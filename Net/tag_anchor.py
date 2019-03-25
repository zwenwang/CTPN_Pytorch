# coding=utf-8
import numpy as np
import math


def cal_IoU(cy1, h1, cy2, h2):
    """
    算IoU的，输入两个anchor框的中心点和高度
    :param cy1: 第一个中心点
    :param h1: 第一个高度
    :param cy2: 第二个中心点
    :param h2: 第二个高度
    :return: IoU
    """
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    cross_h = min(y_bottom1, y_bottom2) - max(y_top1, y_top2) + 1
    if cross_h <= 0:
        return 0.0
    else:
        return float(cross_h) / float(h1 + h2 - cross_h)


# def cal_IoU_old(cy1, h1, cy2, h2):
#     y_top1, y_bottom1 = cal_y(cy1, h1)
#     y_top2, y_bottom2 = cal_y(cy2, h2)
#     offset = min(y_top1, y_top2)
#     y_top1 = y_top1 - offset
#     y_top2 = y_top2 - offset
#     y_bottom1 = y_bottom1 - offset
#     y_bottom2 = y_bottom2 - offset
#     line = np.zeros(max(y_bottom1, y_bottom2) + 1)
#     for i in range(y_top1, y_bottom1 + 1):
#         line[i] += 1
#     for j in range(y_top2, y_bottom2 + 1):
#         line[j] += 1
#     union = np.count_nonzero(line, 0)
#     intersection = line[line == 2].size
#     return float(intersection)/float(union)


# 根据中心点和高度算上下y坐标的
def cal_y(cy, h):
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_bottom


# 验证anchor是不是超过边缘的
def valid_anchor(cy, h, height):
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    if bottom > (height * 16 - 1):
        return False
    return True


def tag_anchor(data, height, width):
    """
    把每个anchor打标记的，就是看他是正样本，负样本，还是算回归的
    :param data: 从数据库里读出的数据
    :param height: cnn输出的feature map高度
    :param width: feature map宽度
    :return: 返回四个列表，里面都是标记好的anchor
    """
    gt_box = data[0]
    gt_anchor = data[1]
    # anchor_num = data[2]
    # 论文里每个anchor1的高度
    anchor_height_list = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]
    # feature map的尺寸
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []
    # 看看每个gt框的左右坐标
    x_left_side = min(gt_box[0], gt_box[6])
    x_right_side = max(gt_box[2], gt_box[4])
    left_side = False
    right_side = False
    for anchor in gt_anchor:
        # 看看每个anchor的位置是不是超过特征图的边缘了
        if anchor[0] >= int(width - 1):
            continue
        # 看看anchor是不是在文本框的边缘
        if x_left_side in range(anchor[0] * 16, (anchor[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        if x_right_side in range(anchor[0] * 16, (anchor[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        # 这一条16px每个高度搞一个iou
        iou = np.zeros((height, len(anchor_height_list)))
        temp_positive = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                # 看看feature map上的anchor是不是超出边界之类的
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height_list[j], height):
                    continue
                # 算IoU
                # TODO: 有一些小问题，但是对最终结果基本没有影响
                iou[i][j] = cal_IoU((float(i) * 16.0 + 7.5), anchor_height_list[j], anchor[1], anchor[2])

                # 大于0.7的正样本
                if iou[i][j] > 0.7:
                    # 横向位置，纵向位置，第几个高度，iou
                    temp_positive.append((anchor[0], i, j, iou[i][j]))
                    # 正样本里有在边缘的算side-refinement
                    if left_side:
                        o = (float(x_left_side) - (float(anchor[0]) * 16.0 + 7.5)) / 16.0
                        # 横向位置，纵向位置，第几个高度，算side-refinement的o
                        side_refinement_reg.append((anchor[0], i, j, o))
                    if right_side:
                        o = (float(x_right_side) - (float(anchor[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((anchor[0], i, j, o))

                # 小于0.5的负样本
                if iou[i][j] < 0.5:
                    # 格式和正样本一样
                    negative.append((anchor[0], i, j, iou[i][j]))

                # 大于0.5的算回归
                if iou[i][j] > 0.5:
                    vc = (anchor[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height_list[j])
                    vh = math.log10(float(anchor[2]) / float(anchor_height_list[j]))
                    # 横向位置，纵向位置，第几个高度，vc，vh
                    vertical_reg.append((anchor[0], i, j, vc, vh, iou[i][j]))

        # 要是没有大于0.7的，就选IoU最高的当正样本
        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((anchor[0], max_position[0][0], max_position[1][0], np.max(iou)))

            if left_side:
                o = (float(x_left_side) - (float(anchor[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((anchor[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(x_right_side) - (float(anchor[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((anchor[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (anchor[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height_list[max_position[1][0]])
                vh = math.log10(float(anchor[2]) / float(anchor_height_list[max_position[1][0]]))

                vertical_reg.append((anchor[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))
        positive += temp_positive
    return positive, negative, vertical_reg, side_refinement_reg
