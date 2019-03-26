# coding=utf-8
import lmdb
import cv2
import other
import codecs
import json
from .generate_gt_anchor import generate_gt_anchor


def get_json_str(img, img_name, gt_box, anchor_width=16):
    # json字符串格式：
    # {
    #   'file' : 图片名称
    #   'data' : [
    #              box的8个点,
    #              存有anchor的列表,
    #              anchor个数
    #            ]
    #  }
    json_obj = {'file': img_name}
    data = []
    for box in gt_box:
        gt_anchor = generate_gt_anchor(img, box, anchor_width=anchor_width)
        if len(gt_anchor) == 0:
            continue
        temp = [box]
        temp.append(gt_anchor)
        temp.append(len(gt_anchor))
        data.append(temp)
    json_obj.update(data=data)
    str_json = json.dumps(json_obj)
    return str_json


def scale_img(img, gt, shortest_side=600, anchor_width=16):
    """
    先把最短边缩放到600,然后将宽度缩放到16的倍数来满足最终的总步长
    :param img: 图片，opencv读取
    :param gt: groundtruth坐标
    :param shortest_side: 最短边长
    :param anchor_width: anchor宽度
    :return: 返回缩放后的图片和坐标
    """
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    # if img.shape[0] < img.shape[1] and img.shape[0] != 600:
    #     img = cv2.resize(img, (600, img.shape[1]))
    # elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
    #     img = cv2.resize(img, (img.shape[0], 600))
    # elif img.shape[0] != 600:
    #     img = cv2.resize(img, (600, 600))
    remainder = img.shape[1] % anchor_width
    img = cv2.resize(img, (img.shape[1] + (anchor_width - remainder), img.shape[0]))
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    if gt is None:
        return img
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    return img, scale_gt


def read_gt_file(path, have_BOM=False):
    """
    读取groundtruth文件，每个框由8个点构成
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
        box = [int(pt[i]) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def check_img(img):
    if img is None:
        return False
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


# def write_cache(env, data):
#     with env.begin(write=True) as e:
#         for i, l in data.iteritems():
#             e.put(i, l)


# def create_dataset(output_path, img_list, gt_list):
#     assert len(img_list) == len(gt_list)
#     num = len(img_list)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     env = lmdb.open(output_path, map_size=1099511627776)
#     cache = {}
#     counter = 1
#     for i in range(num):
#         img_path = img_list[i]
#         gt_path = gt_list[i]
#         if not os.path.exists(img_path):
#             print("{0} is not exist.".format(img_path))
#             continue
#
#         img = cv2.imread(img_path)
#         if not check_img(img):
#             print('Image {0} is not valid.'.format(img_path))
#             continue
#
#         img_key = 'image-%09d' % counter
#         gt_key = 'gt-%09d' % counter
#         cache[img_key] = other.np_img2base64(img, img_path)
#         cache[gt_key] = gt
#         counter += 1
#         if counter % 100 == 0:
#             write_cache(env, cache)
#             cache.clear()
#             print('Written {0}/{1}'.format(counter, num))
#     cache['num'] = str(counter - 1)
#     write_cache(env, cache)
#     print('Create dataset with {0} image.'.format(counter - 1))


class LMDB():
    def __init__(self, path):
        self.path = path
        self.loaded = False
        self.env = None

    def load(self):
        self.env = lmdb.Environment(self.path)
        self.loaded = True

    def create(self, map_size=1099511627776):
        self.env = lmdb.open(self.path, map_size=map_size)
        txn = self.env.begin(write=True)
        txn.put(key='num', value=str(0))
        txn.commit()
        self.loaded = True

    def insert(self, img, gt, img_name, anchor_width=16):
        """
        向数据库里插入图片
        :param img: 图像，opencv读取的
        :param gt: groundtruth， 格式n×8
        :param img_name: 图像名称，带后缀名的
        :param anchor_width: anchor宽度
        :return:
        """
        if not self.loaded:
            print('Please load or create lmdb first.')
        if not check_img(img):
            print('Image is not valid.')
            return False
        if len(gt) == 0:
            print('GroundTruth is not valid.')
            return False
        img, gt = scale_img(img, gt)
        json_str = get_json_str(img, img_name,  gt, anchor_width=anchor_width)
        txn = self.env.begin(write=True)
        num = int(txn.get('num'))

        img_index = 'image-%09d' % num
        txn.put(key=img_index, value=other.np_img2base64(img, img_name))
        gt_index = 'gt-%09d' % num
        txn.put(key=gt_index, value=json_str)
        txn.put(key='num', value=str(num + 1))
        txn.commit()
        return True

    def query(self, index):
        if not self.loaded:
            print('Please load or create lmdb first.')
        txn = self.env.begin(write=False)
        img_index = 'image-%09d' % index
        gt_index = 'gt-%09d' % index
        img = txn.get(img_index)
        gt = txn.get(gt_index)
        return other.base642np_image(img), json.loads(gt)

    def sum(self):
        if not self.loaded:
            print('Please load or create lmdb first.')
        txn = self.env.begin(write=False)
        num = int(txn.get('num'))
        return num
