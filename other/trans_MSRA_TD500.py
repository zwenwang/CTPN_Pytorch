import math
import cv2
import os
import utils


# Deprecated
def get_box_img(x, y, w, h, angle):
    x0 = x + w/2
    y0 = y + h/2
    l = math.sqrt(pow(w/2, 2) + pow(h/2, 2))
    if angle < 0:
        a1 = -angle + math.atan(h / float(w))
        a2 = -angle - math.atan(h / float(w))
        pt1 = (x0 - l * math.cos(a2), y0 + l * math.sin(a2))
        pt2 = (x0 + l * math.cos(a1), y0 - l * math.sin(a1))
        pt3 = (x0 + l * math.cos(a2), y0 - l * math.sin(a2))
        pt4 = (x0 - l * math.cos(a1), y0 + l * math.sin(a1))
    else:
        a1 = angle + math.atan(h / float(w))
        a2 = angle - math.atan(h / float(w))
        pt1 = (x0 - l * math.cos(a1), y0 - l * math.sin(a1))
        pt2 = (x0 + l * math.cos(a2), y0 + l * math.sin(a2))
        pt3 = (x0 + l * math.cos(a1), y0 + l * math.sin(a1))
        pt4 = (x0 - l * math.cos(a2), y0 - l * math.sin(a2))
    return [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]


def read_file(path):
    result = []
    for line in open(path):
        info = []
        data = line.split(' ')
        info.append(int(data[2]))
        info.append(int(data[3]))
        info.append(int(data[4]))
        info.append(int(data[5]))
        info.append(float(data[6]))
        info.append(data[0])
        result.append(info)
    return result


if __name__ == '__main__':
    file_path = './MSRA-TD500/test/'
    save_img_path = './MSRA-dataset/test_img/'
    save_gt_path = './MSRA-dataset/test_gt/'
    file_list = os.listdir(file_path)
    for f in file_list:
        if '.gt' in f:
            continue
        name = f[0:8]
        txt_path = file_path + name + '.gt'
        im_path = file_path + f
        im = cv2.imread(im_path)
        coordinate = read_file(txt_path)

        cv2.imwrite(save_img_path + name.lower() + '.jpg', im)
        save_gt = open(save_gt_path + 'gt_' + name.lower() + '.txt', 'w')
        for i in coordinate:
            box = get_box_img(i[0], i[1], i[2], i[3], i[4])
            box = [int(box[i]) for i in range(len(box))]
            box = [str(box[i]) for i in range(len(box))]
            save_gt.write(','.join(box))
            save_gt.write('\n')

