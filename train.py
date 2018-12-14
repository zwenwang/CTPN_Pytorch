# coding=utf-8
import torch.optim as optim
import torch
import Dataset
import Net
import os
import other
import ConfigParser
import time
import val_func
import logging
import datetime


if __name__ == '__main__':
    # 读配置文件
    cf = ConfigParser.ConfigParser()
    cf.read('./config')

    # 创建日志文件
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

    # 读一些配置
    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')
    logger.info('Total epoch: {0}'.format(epoch))

    using_cuda = cf.getboolean('global', 'using_cuda')
    display_iter = cf.getint('global', 'display_iter')
    val_iter = cf.getint('global', 'val_iter')
    save_iter = cf.getint('global', 'save_iter')

    optimizer_type = cf.get('parameter', 'optimizer')
    lr = cf.getfloat('parameter', 'lr')

    pretrained = cf.getboolean('global', 'pretrained')
    pretrained_model = cf.get('global', 'pretrained_model')

    batch_size = cf.getint('global', 'batch_size')
    sample_ratio = cf.getfloat('global', 'sample_ratio')
    test_batch_num = cf.getint('global', 'test_batch_num')

    have_prefix = cf.getboolean('global', 'have_prefix')
    prefix = cf.get('global', 'prefix')

    # 指定不需要更新参数的层
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = [
        'cnn.VGG_16.convolution1_1.weight',
        'cnn.VGG_16.convolution1_1.bias',
        'cnn.VGG_16.convolution1_2.weight',
        'cnn.VGG_16.convolution1_2.bias'
    ]

    # 初始化网络，根据选项载入预训练的VGG模型或者检查点
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

    criterion = Net.CTPN_Loss(batch_size, sample_ratio, using_cuda=using_cuda)

    # 使用CUDA
    if using_cuda:
        net.cuda()
        criterion = criterion.cuda()
    net.train()
    print(net)
    print('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    logger.info('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    print('Train epoch: {0}'.format(epoch))
    logger.info('Train epoch: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))
    logger.info('Use CUDA: {0}'.format(using_cuda))

    # 用torch里的东西来读数据
    train_dataset = Dataset.LmdbDataset(cf.get('global', 'train_dataset'))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = Dataset.LmdbDataset(cf.get('global', 'test_dataset'))

    if optimizer_type == 'SGD':
        momentum = cf.getfloat('parameter', 'momentum')
        weight_decay = cf.getfloat('parameter', 'weight_decay')
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        print('Use SGD, lr={0}, momentum={1}, weight decay={2}'.format(lr, momentum, weight_decay))
        logger.info('Use SGD, lr={0}, momentum={1}, weight decay={2}'.format(lr, momentum, weight_decay))

    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
        print('Use Adam, lr={0}'.format(lr))
        logger.info('Use Adam, lr={0}'.format(lr))

    elif optimizer_type == 'Adadelta':
        rho = cf.getfloat('parameter', 'rho')
        optimizer = optim.Adadelta(net.parameters(), lr=lr, rho=rho)
        print('Use Adadelta, lr={0}, rho={1}'.format(lr, rho))
        logger.info('Use Adadelta, lr={0}, rho={1}'.format(lr, rho))

    for i in range(epoch):
        iteration = 1
        total_loss = 0
        total_cls_loss = 0
        total_v_reg_loss = 0
        total_o_reg_loss = 0
        start_time = time.time()
        train_iter = iter(train_loader)
        total_iter = len(train_loader)
        for j in range(total_iter):
            data = train_iter.next()
            img, gt = data
            tensor_img = img.transpose(1, 3)
            tensor_img = tensor_img.transpose(2, 3)
            tensor_img = tensor_img.float()
            if using_cuda:
                tensor_img = tensor_img.cuda()

            # 将图片送入网络并产生结果
            vertical_pred, score, side_refinement = net(tensor_img)
            # 总是显存爆炸，所以删了图片的tensor
            del tensor_img
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

            # 清梯度，算loss，反传
            net.zero_grad()
            loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                               negative, vertical_reg, side_refinement_reg)
            loss.backward()
            optimizer.step()
            iteration += 1
            total_loss += loss
            total_cls_loss += cls_loss
            total_v_reg_loss += v_reg_loss
            total_o_reg_loss += o_reg_loss

            # 显示
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
                print('****************************************************')
                total_loss = 0
                total_cls_loss = 0
                total_v_reg_loss = 0
                total_o_reg_loss = 0
                start_time = time.time()

            # 验证
            if iteration % val_iter == 0:
                net.eval()
                logger.info('Start evaluate at {0} epoch {1} iteration.'.format(i, iteration))
                val_func.val(net, criterion, test_batch_num, using_cuda, logger, test_dataset)
                logger.info('End evaluate.')
                net.train()
                start_time = time.time()

            if iteration % save_iter == 0:
                if have_prefix:
                    print('Model saved at ./model/{2}-ctpn-{0}-{1}.model'.format(i, iteration, prefix))
                    torch.save(net.state_dict(), './{2}-model/ctpn-{0}-{1}.model'.format(i, iteration, prefix))
                else:
                    print('Model saved at ./model/ctpn-{0}-{1}.model'.format(i, iteration))
                    torch.save(net.state_dict(), './model/ctpn-{0}-{1}.model'.format(i, iteration))

        # 每个epoch完事儿存一下
        if have_prefix:
            print('Model saved at ./model/{1}-ctpn-{0}-end.model'.format(i, prefix))
            torch.save(net.state_dict(), './model/{1}-ctpn-{0}-end.model'.format(i, prefix))
        else:
            print('Model saved at ./model/ctpn-{0}-end.model'.format(i))
            torch.save(net.state_dict(), './model/ctpn-{0}-end.model'.format(i))
