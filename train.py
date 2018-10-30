import torch.optim as optim
import torch
import cv2
import Dataset.port
import other
import Net
import numpy as np
from torch.autograd import Variable


if __name__ == '__main__':
    net = Net.CTPN()
    for p in net.parameters():
        p.requires_grad = True
    net.train()
    print(net)
    img = cv2.imread('./img_112.jpg')
    gt_txt = Dataset.port.read_gt_file('./gt_img_112.txt', have_BOM=True)
    img, gt_txt = Dataset.scale_img(img, gt_txt)
    tensor_img = img[np.newaxis, :, :, :]
    tensor_img = tensor_img.transpose((0, 3, 1, 2))
    tensor_img = torch.FloatTensor(tensor_img)
    vertical_pred, score, side_refinement = net(tensor_img)
    positive = []
    negative = []
    vertical_reg = []
    for box in gt_txt:
        gt_anchor = Dataset.generate_gt_anchor(img, box)
        positive1, negative1, vertical_reg1 = Net.tag_anchor(gt_anchor, score, box)
        positive += positive1
        negative += negative1
        vertical_reg += vertical_reg1
    criterion = Net.CTPN_Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()
    loss = criterion(score, vertical_pred, side_refinement, positive, negative, vertical_reg)
    loss.backward()
    optimizer.step()
    print(loss)
