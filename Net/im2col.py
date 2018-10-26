import numpy as np
import copy


def im2col(raw_im, kernel_size, padding, stride):
    """
    :param im: Pytorch tensor, shape batch_size * channel * height * width
    :param kernel_size: kernel size
    :param padding: (height_padding. width_padding)
    :param stride: (height_stride, width_stride)
    :return:
    """
    im = copy.deepcopy(raw_im)
    batch_size, channel, height, width = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
    if padding is not None:
        im = np.pad(im, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant')
    result = np.zeros((batch_size, kernel_size[0] * kernel_size[1] * channel,
                       ))
    return im
