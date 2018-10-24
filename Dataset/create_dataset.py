import os
import lmdb
import numpy as np
import os
import cv2


def check_img(raw_data):
    if raw_data is None:
        return False
    buf = np.fromstring(raw_data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


def write_cache(env, data):
    with env.begin(write=True) as e:
        for i, l in data.iteritems():
            e.put(i, l)


def list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True


def create_dataset(output_path, img_list, gt_list):
    assert len(img_list) == len(gt_list)
    num = len(img_list)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    counter = 1
    for i in range(num):
        img_path = img_list[i]
        gt = gt_list[i]
        if not os.path.exists(img_path):
            print("{0} is not exist.".format(img_path))
            continue

        if len(gt) == 0:
            print("Ground truth of {0} is not exist.".format(img_path))
            continue

        with open(img_path, 'r') as f:
            raw_data = f.read()
        if not check_img(raw_data):
            print('Image {0} is not valid.'.format(img_path))
            continue

        gt_str = list2str(gt)
        if not gt_str[1]:
            print("Ground truth of {0} is not valid.".format(img_path))
            continue

        img_key = 'image-%09d' % counter
        gt_key = 'gt-%09d' % counter
        cache[img_key] = raw_data
        cache[gt_key] = gt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))
        cache['num'] = str(counter - 1)
