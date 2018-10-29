import numpy as np


def cal_IoU(cy1, h1, cy2, h2):
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    offset = min(y_top1, y_top2)
    y_top1 = y_top1 - offset
    y_top2 = y_top2 - offset
    y_bottom1 = y_bottom1 - offset
    y_bottom2 = y_bottom2 - offset
    line = np.zeros(max(y_bottom1, y_bottom2) + 1)
    for i in range(y_top1, y_bottom1 + 1):
        line[i] += 1
    for j in range(y_top2, y_bottom2 + 1):
        line[j] += 1
    union = np.count_nonzero(line, 0)
    intersection = line[line == 2].size
    return float(intersection)/float(union)


def cal_y(cy, h):
    if h % 2 == 0:
        y_bottom = cy + h / 2
        y_top = y_bottom - h + 1
    else:
        y_top = cy - (h - 1) / 2
        y_bottom = cy + (h - 1) / 2
    return int(y_top), int(y_bottom)


def tag_anchor(gt_anchor, cnn_output, gt_box):
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]
    height = cnn_output[2]
    width = cnn_output[3]
    positive = {'position': [], 'vc': [], 'vh': [], 'o': []}
    negative = {'position': [], 'vc': [], 'vh': [], 'o': []}
    x_left_side = min(gt_box[0], gt_box[6])
    x_right_side = max(gt_box[2], gt_box[4])
    for p in gt_anchor['position']:
        if p > (width - 1):
            continue
        iou = np.zeros((height, len(anchor_height)))
