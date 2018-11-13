import numpy as np
cimport numpy as np


def find_top_bottom(np.ndarray[np.uint8_t, ndim=3] img, list position_pair):
    cdef list y_top = []
    cdef list y_bottom = []
    cdef int height = img.shape[0]
    cdef int top_flag = 0
    cdef int bottom_flag = 0
    cdef int k, x, y
    for k in range(len(position_pair)):
        for y in range(0, height):
            for x in range(position_pair[k][0], position_pair[k][1] + 1):
                if img[y, x, 0] == 255:
                    y_top.append(y)
                    top_flag = 1
                    break
            if top_flag == 1:
                break
        for y in range(height - 1, -1, -1):
            for x in range(position_pair[k][0], position_pair[k][1] + 1):
                if img[y, x, 0] == 255:
                    y_bottom.append(y)
                    bottom_flag = 1
                    break
            if bottom_flag == 1:
                break
        top_flag = 0
        bottom_flag = 0
    return y_top, y_bottom