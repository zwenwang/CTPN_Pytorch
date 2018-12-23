# coding=utf-8
import torch
import Net
import time


def val(net, criterion, batch_num, using_cuda, logger, test_dataset):
    print('####################  Start evaluate  ####################')
    total_loss = 0.0
    total_cls_loss = 0.0
    total_v_reg_loss = 0.0
    total_o_reg_loss = 0.0
    start_time = time.time()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_iter = iter(test_loader)
    test_num = min(len(test_loader), batch_num)
    for i in range(test_num):
        data = test_iter.next()
        img, gt = data
        img = img.transpose(1, 3)
        img = img.transpose(2, 3)
        img = img.float()
        if using_cuda:
            img = img.cuda()

        # 将图片送入网络并产生结果
        vertical_pred, score, side_refinement = net(img)
        # 总是显存爆炸，所以删了图片的tensor
        del img
        # 用来存真实值的
        positive = []
        negative = []
        vertical_reg = []
        side_refinement_reg = []
        for box in gt['data']:
            # 根据分好的anchor产生每个输出要的anchor
            positive1, negative1, vertical_reg1, side_refinement_reg1 = Net.tag_anchor(box[1], score, box[0])
            positive += positive1
            negative += negative1
            vertical_reg += vertical_reg1
            side_refinement_reg += side_refinement_reg1

        # 算loss
        loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                           negative, vertical_reg, side_refinement_reg)
        total_loss += float(loss)
        total_cls_loss += float(cls_loss)
        total_v_reg_loss += float(v_reg_loss)
        total_o_reg_loss += float(o_reg_loss)

    end_time = time.time()
    total_time = end_time - start_time
    print('loss: {0}'.format(total_loss / float(test_num)))
    logger.info('Evaluate loss: {0}'.format(total_loss / float(test_num)))

    print('classification loss: {0}'.format(total_cls_loss / float(test_num)))
    logger.info('Evaluate vertical regression loss: {0}'.format(total_v_reg_loss / float(test_num)))

    print('vertical regression loss: {0}'.format(total_v_reg_loss / float(test_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(test_num)))

    print('side-refinement regression loss: {0}'.format(total_o_reg_loss / float(test_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(test_num)))

    print('{1} iterations for {0} seconds.'.format(total_time, test_num))
    print('#####################  Evaluate end  #####################')
