import math
import other


def generate_gt_anchor(img, box, anchor_width=16):
    if isinstance(box[0], str):
        box = [float(box[i]) for i in range(len(box))]
    result = {'position':[], 'h':[], 'cy':[], 'o':[]}
    