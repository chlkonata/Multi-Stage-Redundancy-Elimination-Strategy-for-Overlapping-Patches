import json
import os

def compute_iou(box1, box2):
    """
    计算两个检测框的IOU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(box1_area + box2_area - intersection)
    return iou


def compute_giou(box1, box2):
    """
    计算两个检测框的 GIoU
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    # 计算最小包围框
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    return giou


def load_detection_results(file_path, img_w, img_h):
    """
    从YOLO格式文件中加载检测结果

    :param file_path: 文件路径
    :param img_w: 图像宽度
    :param img_h: 图像高度
    :return: 检测框列表[x1, y1, x2, y2, conf, class_id]
    """
    if not os.path.exists(file_path):
        return []

    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            class_id = int(parts[0])
            conf = float(parts[1])
            x_center = float(parts[2]) * img_w
            y_center = float(parts[3]) * img_h
            width = float(parts[4]) * img_w
            height = float(parts[5]) * img_h

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append([x1, y1, x2, y2, conf, class_id])

    return boxes
