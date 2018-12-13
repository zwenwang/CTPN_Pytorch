import Dataset
import os


def create_dataset_icdar2015(img_root, gt_root, output_path):
    im_list = os.listdir(img_root)
    im_path_list = []
    gt_list = []
    for im in im_list:
        name, _ = os.path.splitext(im)
        gt_name = 'gt_' + name + '.txt'
        gt_path = os.path.join(gt_root, gt_name)
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(im))
        im_path_list.append(os.path.join(img_root, im))
        gt_list.append(gt_path)
    assert len(im_path_list) == len(gt_list)
    Dataset.create_dataset(output_path, im_path_list, gt_list)
