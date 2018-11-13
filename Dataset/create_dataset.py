import os
import lmdb
import cv2
import other
import Net


def scale_img(img, gt, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
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


def check_img(img):
    if img is None:
        return False
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


def write_cache(env, data):
    with env.begin(write=True) as e:
        for i, l in data.iteritems():
            e.put(i, l)


def box_list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True


def create_dataset(output_path, img_list, gt_list):
    assert len(img_list) == len(gt_list)
    net = Net.VGG_16()
    num = len(img_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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

        img = cv2.imread(img_path)
        if not check_img(img):
            print('Image {0} is not valid.'.format(img_path))
            continue

        img, gt = scale_img(img, gt)
        gt_str = box_list2str(gt)
        if not gt_str[1]:
            print("Ground truth of {0} is not valid.".format(img_path))
            continue

        img_key = 'image-%09d' % counter
        gt_key = 'gt-%09d' % counter
        cache[img_key] = other.np_img2base64(img, img_path)
        cache[gt_key] = gt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))
    cache['num'] = str(counter - 1)
    write_cache(env, cache)
    print('Create dataset with {0} image.'.format(counter - 1))
