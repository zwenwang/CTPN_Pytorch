# --coding:utf8--
import numpy as np


MAX_HORIZONTAL_GAP = 50
MIN_V_OVERLAPS = 0.7
MIN_SIZE_SIM = 0.7


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    # 将两个文本框合成为一个文本框对
    # 方法：遍历所有文本框，在所有文本框后面50px之内观察有没有其余文本框
    # 如果有，那么根据论文中的判断方法，确定是否可以生成一对文本框
    def __init__(self):
        pass

    def get_successions(self, index):  # 生成文本框对
            box = self.text_proposals[index]  # 先根据index从怕人哦破萨拉里选一个
            results = []
            for left in range(int(box[0])+1, min(int(box[0]) + MAX_HORIZONTAL_GAP + 1, self.im_size[1])):  # 根据box的第0个横坐标，查找坐标右边50px内的所有proposal
                # print(left)
                adj_box_indices = self.boxes_table[left]  # 根据每个px从boxes tabel里找对应的列表，即图片的每一列一个列表
                for adj_box_index in adj_box_indices:  # 从这列里遍历所有的proposal
                    if self.meet_v_iou(adj_box_index, index):  # 满足一定条件的proposal放在result里
                        results.append(adj_box_index)
                if len(results) != 0:
                    return results
            return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0])-1, max(int(box[0] - MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2) >= MIN_V_OVERLAPS and size_similarity(index1, index2) >= MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1  # 计算每个proposal的高度

        # print(text_proposals)
        # print(self.heights)
        # print(text_proposals)

        boxes_table = [[] for _ in range(self.im_size[1])]  # 创建了一个长800的列表，根据图片宽度
        for index, box in enumerate(text_proposals):  # 根据box的第0个坐标，对应到boxtables里，并且统计了其坐标
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table
        # print(boxes_table)

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)  # 创建了一个bool图，大小根据proposal数量见方

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)
