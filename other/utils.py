import cv2
import numpy as np
import base64
import os
import torch
import math
from matplotlib import cm


def draw_box_4pt(img, pt, color=(0, 255, 0), thickness=1):
    assert len(pt) == 8
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(8)]
    img = cv2.line(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness)
    img = cv2.line(img, (pt[2], pt[3]), (pt[4], pt[5]), color, thickness)
    img = cv2.line(img, (pt[4], pt[5]), (pt[6], pt[7]), color, thickness)
    img = cv2.line(img, (pt[6], pt[7]), (pt[0], pt[1]), color, thickness)
    return img


def draw_box_2pt(img, pt, color=(0, 255, 0), thickness=1):
    assert len(pt) == 4
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(4)]
    img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness=thickness)
    return img


def draw_box_h_and_c(img, position, cy, h, anchor_width=16, color=(0, 255, 0), thickness=1):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    pt = [x_left, y_top, x_right, y_bottom]
    return draw_box_2pt(img, pt, color=color, thickness=thickness)


def draw_boxes(im, bboxes, is_display=True, color=None, thickness=1):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)

    im = im.copy()
    index = 0
    for box in bboxes:
        if color is None:
            if len(box) == 8 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), c, thickness=thickness)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), c, thickness=thickness)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), c, thickness=thickness)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), c, thickness=thickness)
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x4
        text_recs[index, 5] = y4
        text_recs[index, 6] = x3
        text_recs[index, 7] = y3
        index = index + 1
        # cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)
    if is_display:
        cv2.imshow('result', im)
        cv2.waitKey(0)
    return text_recs


def trans_to_2pt(position, cy, h, anchor_width=16):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return [x_left, y_top, x_right, y_bottom]


def np_img2base64(np_img, path):
    image = cv2.imencode(os.path.splitext(path)[1], np_img)[1]
    image = np.squeeze(image, 1)
    image_code = base64.b64encode(image)
    return image_code


def base642np_image(base64_str):
    missing_padding = 4 - len(base64_str) % 4
    if missing_padding:
        base64_str += b'=' * missing_padding
    raw_str = base64.b64decode(base64_str)
    np_img = np.fromstring(raw_str, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.COLOR_RGB2BGR)
    return img


def cal_line_y(pt1, pt2, x, form):
    if not isinstance(pt1[0], float) or not isinstance(pt2[0], float):
        pt1 = [float(pt1[i]) for i in range(len(pt1))]
        pt2 = [float(pt2[i]) for i in range(len(pt2))]
    if not isinstance(x, float):
        x = float(x)
    if (pt1[0] - pt2[0]) == 0:
        return -1
    return form(((pt1[1] - pt2[1])/(pt1[0] - pt2[0])) * (x - pt1[0]) + pt1[1])


def bi_range(start, end):
    start = int(start)
    end = int(end)
    if start > end:
        return range(end, start)
    else:
        return range(start, end)


def init_weight(net):
    for i in range(len(net.rnn.blstm.lstm.all_weights)):
        for j in range(len(net.rnn.blstm.lstm.all_weights[0])):
            torch.nn.init.normal_(net.rnn.blstm.lstm.all_weights[i][j], std=0.01)

    torch.nn.init.normal_(net.FC.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.FC.bias, val=0)

    torch.nn.init.normal_(net.vertical_coordinate.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.vertical_coordinate.bias, val=0)

    torch.nn.init.normal_(net.score.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.score.bias, val=0)

    torch.nn.init.normal_(net.side_refinement.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.side_refinement.bias, val=0)


def perspective_trans(src, img, mode=max):
    distance = lambda x_1, y_1, x_2, y_2: math.sqrt(math.pow(x_1 - x_2, 2) + math.pow(y_1 - y_2, 2))
    width = int(mode(distance(src[0], src[1], src[2], src[3]), distance(src[6], src[7], src[4], src[5])))
    height = int(mode(distance(src[0], src[1], src[6], src[7]), distance(src[2], src[3], src[4], src[5])))

    m = cv2.getPerspectiveTransform(np.float32([src[0:2], src[2:4], src[4:6], src[6:]]),
                                    np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
    result = cv2.warpPerspective(img, m, (width, height))
    return result


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data-min_)/(max_-min_) if max_-min_ != 0 else data-min_


def img_slicing(img, max_height, max_width):
    height, width, channel = img.shape
    col = math.ceil(float(width) / float(max_width))
    row = math.ceil(float(height) / float(max_height))
    block_width = int(round(float(width) / col))
    block_height = int(round(float(height) / row))
    img = cv2.resize(img, (int(block_width * col), int(block_height * row)))
    img_series = []
    for i in range(int(row)):
        for j in range(int(col)):
            img_series.append(img[i * block_height:(i + 1) * block_height - 1,
                              j * block_width:(j + 1) * block_width - 1, :])
    name = '2'
    for i in img_series:
        cv2.imwrite(name + '.jpg', i)
        name = name + '1'
    return np.array(img_series)
