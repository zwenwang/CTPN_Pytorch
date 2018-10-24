import math
import other


def generate_gt_anchor(img, box, anchor_width=16):
    if isinstance(box[0], str):
        box = [float(box[i]) for i in range(len(box))]
    result = {}
    result['position'] = []
    result['h'] = []
    result['cy'] = []
    left_side = abs(int((box[0] + box[6]) / 2.0))
    right_side = abs(int((box[2] + box[4]) / 2.0))
    anchor_x_location = range(0, img.shape[1], anchor_width)
    left_anchor_position = int(math.floor(float(left_side) / float(anchor_width)))
    right_anchor_position = int(math.ceil(float(right_side) / float(anchor_width)))

    y_top = int((box[1] + other.cal_line_y([box[0], box[1]], [box[2], box[3]], left_anchor_position * 16 + 15) / 2))
    # print(anchor_x_location)
    # print(box)
    # print(left_side)
    # print(left_anchor_position)
    # print(right_side)
    # print(right_anchor_position)
