import other


# Dataset.generate_anchor for calculate crossover point between
# gt box and a x-axis
def cal_y_crossover_pt(box, x):
    result = {'y': [], 'edge': []}
    pt1 = [box[0], box[1]]
    pt2 = [box[2], box[3]]
    pt3 = [box[4], box[5]]
    pt4 = [box[6], box[7]]
    y1 = other.cal_line_y(pt1, pt2, x, int)
    y2 = other.cal_line_y(pt2, pt3, x, int)
    y3 = other.cal_line_y(pt3, pt4, x, int)
    y4 = other.cal_line_y(pt4, pt1, x, int)
    if y1 in other.bi_range(pt1[1], pt2[1]):
        result['y'].append(y1)
        result['edge'].append(1)
    if y2 in other.bi_range(pt2[1], pt3[1]):
        result['y'].append(y2)
        result['edge'].append(2)
    if y3 in other.bi_range(pt3[1], pt4[1]):
        result['y'].append(y3)
        result['edge'].append(3)
    if y4 in other.bi_range(pt4[1], pt1[1]):
        result['y'].append(y4)
        result['edge'].append(4)
    return result


def box_list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True
