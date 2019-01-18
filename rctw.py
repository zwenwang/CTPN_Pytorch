# coding=utf-8
from Dataset import LMDB
import os
import cv2
import codecs


def read_gt_file(path, have_BOM=False):
    """
    读取groundtruth文件，每个框由8个点构成,跳过difficult的box
    :param path: 路径
    :param have_BOM: 是否有BOM字符串
    :return: 列表
    """
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        difficult = int(pt[8])
        if difficult == 1:
            continue
        box = [int(pt[i]) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def main():
    part1 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part1'
    part2 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part2'
    part3 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part3'
    part4 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part4'
    part5 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part5'
    part6 = '/media/wzw/587231C57231A920/数据集/RCTW/train/part6'
    train_db = LMDB('/media/wzw/587231C57231A920/数据集/RCTW/LMDB/train')
    train_db.create()
    to_db(part1, train_db)
    to_db(part2, train_db)
    to_db(part3, train_db)
    to_db(part4, train_db)
    to_db(part5, train_db)
    print('Train: {0} image'.format(train_db.sum()))
    test_db = LMDB('/media/wzw/587231C57231A920/数据集/RCTW/LMDB/test')
    test_db.create()
    to_db(part6, test_db)
    print('Test: {0} image'.format(test_db.sum()))


def to_db(root, db):
    file_list = os.listdir(root)
    for file_name in file_list:
        if '.jpg' in file_name:
            img = cv2.imread(os.path.join(root, file_name))
            gt_name = file_name.split('.')[0] + '.txt'
            gt = read_gt_file(os.path.join(root, gt_name))
            if db.insert(img, gt, file_name):
                print('Insert image {0} success!'.format(file_name))
            else:
                print('\033[0;31mInsert image {0} fail!\033[0m'.format(file_name))
    print(db.sum())


if __name__ == '__main__':
    main()
