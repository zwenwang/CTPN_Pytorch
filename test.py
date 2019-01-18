import cv2
import numpy as np
import other
import copy
import Dataset
import torch
import Net
from other.lib import nms
import math
from proposal_connector import TextProposalConnector
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]


MIN_RATIO = 1.2
LINE_MIN_SCORE = 0.7
TEXT_PROPOSALS_WIDTH = 16
MIN_NUM_PROPOSALS = 2
TEXT_LINE_NMS_THRESH = 0.3


def filter_boxes(boxes):
    heights = boxes[:, 3] - boxes[:, 1] + 1
    widths = boxes[:, 2] - boxes[:, 0] + 1
    scores = boxes[:, -1]
    return np.where((widths / heights > MIN_RATIO) & (scores > LINE_MIN_SCORE) &
                    (widths > (TEXT_PROPOSALS_WIDTH * MIN_NUM_PROPOSALS)))[0]


if __name__ == '__main__':
    text_connector = TextProposalConnector()
    net = Net.CTPN()
    net.load_state_dict(torch.load('./model/ctpn-cn.model'))
    net.cuda()
    print(net)
    net.eval()
    # im = cv2.imread('/home/wzw/ICDAR2015/test_image/img_99.jpg')
    im = cv2.imread('/home/wzw/IMG_0513.JPG')
    # im = cv2.imread('/home/wzw/990714656.jpg')
    # gt = Dataset.port.read_gt_file('/home/wzw/ICDAR2015/test_gt/gt_img_99.txt')
    im = Dataset.scale_img(im, None, shortest_side=600)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    img = img.cuda()
    v, score, side = net(img, val=True)
    score = score.cpu().detach().numpy()[:, :, :, 1]
    result = np.where(score > 0.7)
    for_nms = []
    for anchor, height, width in zip(result[0], result[1], result[2]):
        vc = v[anchor, 0, height, width]
        vh = v[anchor, 1, height, width]
        cya = height * 16 + 7.5
        ha = anchor_height[anchor]
        cy = vc * ha + cya
        h = math.pow(10, vh) * ha
        pt = other.trans_to_2pt(width, cy, h)
        for_nms.append([pt[0], pt[1], pt[2], pt[3], score[anchor, height, width]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = nms.cpu_nms(for_nms, TEXT_LINE_NMS_THRESH)
    text_proposals = []
    text_proposal_score = []
    for i in nms_result:
        text_proposals.append(for_nms[i, 0:4])
        text_proposal_score.append(for_nms[i, 4])
    text_proposals = np.array(text_proposals)
    text_proposal_score = np.array(text_proposal_score)
    text_proposal_score = other.normalize(text_proposal_score)
    text_lines = text_connector.get_text_lines(text_proposals, text_proposal_score, im.shape[:2])

    keep_index = filter_boxes(text_lines)
    text_lines = text_lines[keep_index]

    # nms for text lines
    if text_lines.shape[0] != 0:
        keep_inds = nms.cpu_nms(text_lines, TEXT_LINE_NMS_THRESH)
        text_lines = text_lines[keep_inds]

    rec = other.draw_boxes(im, text_lines)

    # cv2.imshow('kk', im)
    # cv2.waitKey(0)
