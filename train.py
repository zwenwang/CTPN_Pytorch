import torch.optim as optim
import torch
import cv2
import Dataset.port
import Net
import numpy as np
import os
import other
import ConfigParser
import time
import val_func
import logging
import datetime
import copy
import random


if __name__ == '__main__':
    cf = ConfigParser.ConfigParser()
    cf.read('./config')

    log_dir = './logs'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_handler = logging.FileHandler(os.path.join(log_dir, log_file_name), 'w')
    log_format = formatter = logging.Formatter('%(asctime)s: %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')
    logger.info('Total epoch: {0}'.format(epoch))

    using_cuda = cf.getboolean('global', 'using_cuda')
    display_img_name = cf.getboolean('global', 'display_file_name')
    display_iter = cf.getint('global', 'display_iter')
    val_iter = cf.getint('global', 'val_iter')
    save_iter = cf.getint('global', 'save_iter')

    lr_front = cf.getfloat('parameter', 'lr_front')
    lr_behind = cf.getfloat('parameter', 'lr_behind')
    change_epoch = cf.getint('parameter', 'change_epoch') - 1

    pretrained = cf.getboolean('global', 'pretrained')
    pretrained_model = cf.get('global', 'pretrained_model')
    logger.info('Learning rate: {0}, {1}, change epoch: {2}'.format(lr_front, lr_behind, change_epoch + 1))
    print('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    print('Train epoch: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = [
        'cnn.VGG_16.convolution1_1.weight',
        'cnn.VGG_16.convolution1_1.bias',
        'cnn.VGG_16.convolution1_2.weight',
        'cnn.VGG_16.convolution1_2.bias'
    ]

    net = Net.CTPN()
    # for name, value in net.named_parameters():
    #     print('name: {0}, grad: {1}'.format(name, value.requires_grad))

    if pretrained:
        net.load_state_dict(torch.load(pretrained_model))
    else:
        net.load_state_dict(torch.load('./model/vgg16.model'))
        other.init_weight(net)

    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True

    if using_cuda:
        net.cuda()
    net.train()
    print(net)

    criterion = Net.CTPN_Loss(using_cuda=using_cuda)

    img_root = './train_data/train_img'
    gt_root = './train_data/train_gt'

    img_root1 = './train_data/img'
    gt_root1 = './train_data/gt'

    im_list = []
    im_list.append(os.listdir(img_root1))
    im_list.append(os.listdir(img_root))
    total_iter = len(im_list[0]) + len(im_list[1])
    for i in range(epoch):
        if i >= change_epoch:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        iteration = 1
        total_loss = 0
        total_cls_loss = 0
        total_v_reg_loss = 0
        total_o_reg_loss = 0
        start_time = time.time()
        for j in range(len(im_list)):
            random.shuffle(im_list[j])

            for im in im_list[j]:
                name, _ = os.path.splitext(im)
                gt_name = 'gt_' + name + '.txt'
                if j == 1:
                    gt_path = os.path.join(gt_root, gt_name)
                else:
                    gt_path = os.path.join(gt_root1, gt_name)
                if not os.path.exists(gt_path):
                    print('Ground truth file of image {0} not exists.'.format(im))
                    continue

                if j == 1:
                    gt_txt = Dataset.port.read_gt_file(gt_path, have_BOM=True)
                    img = cv2.imread(os.path.join(img_root, im))
                    if display_img_name:
                        print(os.path.join(img_root, im))
                else:
                    gt_txt = Dataset.port.read_gt_file(gt_path)
                    img = cv2.imread(os.path.join(img_root1, im))
                    if display_img_name:
                        print(os.path.join(img_root1, im))
                img, gt_txt = Dataset.scale_img(img, gt_txt)
                tensor_img = img[np.newaxis, :, :, :]
                tensor_img = tensor_img.transpose((0, 3, 1, 2))
                if using_cuda:
                    tensor_img = torch.FloatTensor(tensor_img).cuda()
                else:
                    tensor_img = torch.FloatTensor(tensor_img)

                vertical_pred, score, side_refinement = net(tensor_img)
                del tensor_img
                positive = []
                negative = []
                vertical_reg = []
                side_refinement_reg = []
                for box in gt_txt:
                    gt_anchor = Dataset.generate_gt_anchor(img, box)
                    positive1, negative1, vertical_reg1, side_refinement_reg1 = Net.tag_anchor(gt_anchor, score, box)
                    positive += positive1
                    negative += negative1
                    vertical_reg += vertical_reg1
                    side_refinement_reg += side_refinement_reg1

                optimizer.zero_grad()
                loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                                   negative, vertical_reg, side_refinement_reg)
                loss.backward()
                optimizer.step()
                iteration += 1
                total_loss += loss
                total_cls_loss += cls_loss
                total_v_reg_loss += v_reg_loss
                total_o_reg_loss += o_reg_loss

                if iteration % display_iter == 0:
                    end_time = time.time()
                    total_time = end_time - start_time
                    print('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iteration, total_iter, i, epoch))
                    logger.info('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iteration, total_iter, i, epoch))

                    print('loss: {0}'.format(total_loss / display_iter))
                    logger.info('loss: {0}'.format(total_loss / display_iter))

                    print('classification loss: {0}'.format(total_cls_loss / display_iter))
                    logger.info('classification loss: {0}'.format(total_cls_loss / display_iter))

                    print('vertical regression loss: {0}'.format(total_v_reg_loss / display_iter))
                    logger.info('vertical regression loss: {0}'.format(total_v_reg_loss / display_iter))

                    print('side-refinement regression loss: {0}'.format(total_o_reg_loss / display_iter))
                    logger.info('side-refinement regression loss: {0}'.format(total_o_reg_loss / display_iter))

                    print('{1} iterations for {0} seconds.'.format(int(total_time), display_iter))
                    print('\n')
                    total_loss = 0
                    total_cls_loss = 0
                    total_v_reg_loss = 0
                    total_o_reg_loss = 0
                    start_time = time.time()

                if iteration % val_iter == 0:
                    net.eval()
                    logger.info('Start evaluate at {0} epoch {1} iteration.'.format(i, iteration))
                    val_func.val(net, criterion, 10, using_cuda, logger)
                    logger.info('End evaluate.')
                    net.train()
                    start_time = time.time()

                if iteration % save_iter == 0:
                    print('Model saved at ./model/ctpn-{0}-{1}.model'.format(i, iteration))
                    torch.save(net.state_dict(), './model/ctpn-{0}-{1}.model'.format(i, iteration))

        print('Model saved at ./model/ctpn-{0}-end.model'.format(i))
        torch.save(net.state_dict(), './model/ctpn-{0}-end.model'.format(i))
