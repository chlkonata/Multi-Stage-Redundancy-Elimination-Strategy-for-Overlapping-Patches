import os
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict


def box_iou(box1, box2):
    """计算两组边界框的IoU矩阵"""
    N, M = box1.size(0), box2.size(0)
    if N == 0 or M == 0:
        return torch.zeros(N, M)

    # 计算交集坐标
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 避免除以零
    union = area1[:, None] + area2 - inter
    union[union <= 0] = 1e-6

    return inter / union


def get_image_dimensions(image_path):
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size


def load_detection_results(file_path, img_w, img_h, is_gt=False):
    """加载YOLO格式检测结果"""
    if not os.path.exists(file_path):
        return []

    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < (5 if is_gt else 6):
                continue

            # 解析坐标并转换到像素值
            coords = list(map(float, parts[(1 if is_gt else 2):(5 if is_gt else 6)]))
            x_center = coords[0] * img_w
            y_center = coords[1] * img_h
            width = coords[2] * img_w
            height = coords[3] * img_h

            # 计算边界框坐标
            x1 = max(0, x_center - width / 2)
            y1 = max(0, y_center - height / 2)
            x2 = min(img_w, x_center + width / 2)
            y2 = min(img_h, y_center + height / 2)

            # 只添加有效边界框
            if x2 > x1 and y2 > y1:
                conf = 1.0 if is_gt else float(parts[1])
                class_id = int(parts[0])
                boxes.append([x1, y1, x2, y2, conf, class_id])

    return boxes


def calculate_ap(recall, precision):
    """计算平均精度（AP）"""
    # 将召回率和精确率转换为numpy数组
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # 对精确率进行平滑处理（单调递减）
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # 找到召回率变化的点
    indices = np.where(recall[1:] != recall[:-1])[0] + 1

    # 计算AP（曲线下面积）
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap


def calculate_map_per_class(predictions, targets, iou_thresholds, num_classes=2, max_detections=300):
    """手动计算每个类别的mAP"""
    # 为每个类别和IoU阈值存储预测结果
    class_data = {cls_id: {iou_thresh: {'scores': [], 'tp': [], 'num_gt': 0}
                           for iou_thresh in iou_thresholds}
                  for cls_id in range(num_classes)}

    # 统计每个类别的真实框数量
    for target in targets:
        for cls_id in range(num_classes):
            cls_mask = target['labels'] == cls_id
            class_data[cls_id][iou_thresholds[0]]['num_gt'] += torch.sum(cls_mask).item()

    # 复制真实框数量到所有IoU阈值
    for cls_id in range(num_classes):
        num_gt = class_data[cls_id][iou_thresholds[0]]['num_gt']
        for iou_thresh in iou_thresholds[1:]:
            class_data[cls_id][iou_thresh]['num_gt'] = num_gt

    # 处理每个图像的预测
    for pred, target in zip(predictions, targets):
        # 限制预测框数量
        if len(pred['boxes']) > max_detections:
            # 按置信度排序并选择前max_detections个
            indices = torch.argsort(pred['scores'], descending=True)[:max_detections]
            pred_boxes = pred['boxes'][indices]
            pred_scores = pred['scores'][indices]
            pred_labels = pred['labels'][indices]
        else:
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

        # 为每个类别处理预测
        for cls_id in range(num_classes):
            # 获取该类别的预测
            cls_pred_mask = pred_labels == cls_id
            if not torch.any(cls_pred_mask):
                continue

            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]

            # 获取该类别的真实框
            cls_gt_mask = target['labels'] == cls_id
            cls_gt_boxes = target['boxes'][cls_gt_mask] if torch.any(cls_gt_mask) else torch.zeros(0, 4)

            # 如果没有真实框，所有预测都是FP
            if len(cls_gt_boxes) == 0:
                for iou_thresh in iou_thresholds:
                    class_data[cls_id][iou_thresh]['scores'].extend(cls_pred_scores.tolist())
                    class_data[cls_id][iou_thresh]['tp'].extend([0] * len(cls_pred_scores))
                continue

            # 计算IoU矩阵
            iou_matrix = box_iou(cls_pred_boxes, cls_gt_boxes)

            # 为每个IoU阈值处理
            for iou_thresh in iou_thresholds:
                # 按置信度排序
                sorted_indices = torch.argsort(cls_pred_scores, descending=True)
                sorted_scores = cls_pred_scores[sorted_indices]
                sorted_boxes = cls_pred_boxes[sorted_indices]

                # 跟踪已匹配的真实框
                gt_matched = set()
                tp_list = []

                for i, score in enumerate(sorted_scores):
                    # 找到最佳匹配的真实框
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx in range(len(cls_gt_boxes)):
                        if gt_idx in gt_matched:
                            continue

                        iou_val = iou_matrix[sorted_indices[i], gt_idx].item()
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gt_idx = gt_idx

                    # 判断是否为TP
                    if best_iou >= iou_thresh and best_gt_idx != -1:
                        tp_list.append(1)
                        gt_matched.add(best_gt_idx)
                    else:
                        tp_list.append(0)

                # 存储结果
                class_data[cls_id][iou_thresh]['scores'].extend(sorted_scores.tolist())
                class_data[cls_id][iou_thresh]['tp'].extend(tp_list)

    # 计算每个类别和IoU阈值的AP
    ap_results = {}
    map_results = {}

    for iou_thresh in iou_thresholds:
        aps = []
        for cls_id in range(num_classes):
            data = class_data[cls_id][iou_thresh]
            if not data['scores']:
                ap = 0.0
            else:
                # 按置信度排序
                sorted_indices = np.argsort(-np.array(data['scores']))
                tp = np.array(data['tp'])[sorted_indices]
                scores = np.array(data['scores'])[sorted_indices]

                # 计算累积TP和FP
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(1 - tp)

                # 计算召回率和精确率
                recall = cum_tp / (data['num_gt'] + 1e-6)
                precision = cum_tp / (cum_tp + cum_fp + 1e-6)

                # 计算AP
                ap = calculate_ap(recall, precision)

            aps.append(ap)
            ap_results[f'class{cls_id}_iou{iou_thresh}'] = ap

        # 计算该IoU阈值下的mAP
        map_results[f'mAP_iou{iou_thresh}'] = np.mean(aps) if aps else 0.0

    return ap_results, map_results


def convert_to_target(gt_boxes, width, height, max_detections=None):
    """转换真实标注为目标格式"""
    boxes, labels = [], []
    for box in gt_boxes:
        x1, y1, x2, y2, _, cls_id = box
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1 and cls_id in [0, 1]:  # 只处理类别0和1
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
    labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
    return {"boxes": boxes_tensor, "labels": labels_tensor}


def convert_to_prediction(pred_boxes, width, height, max_detections=300):
    """转换预测结果为预测格式（添加最大检测数量限制）"""
    boxes, scores, labels = [], [], []
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1 and cls_id in [0, 1]:  # 只处理类别0和1
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls_id)

    # 按置信度排序
    if scores:
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        boxes = [boxes[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        labels = [labels[i] for i in sorted_idx]

        # 限制最大检测数量
        if max_detections and len(boxes) > max_detections:
            boxes = boxes[:max_detections]
            scores = scores[:max_detections]
            labels = labels[:max_detections]

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
    scores_tensor = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
    labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

    return {"boxes": boxes_tensor, "scores": scores_tensor, "labels": labels_tensor}


def validation(gt_dir, pred_dir, image_dir, class_map=None,
               output_file="validation_results.json", iou_threshold=0.5,
               max_detections=300):
    """主验证函数"""
    # 加载类别映射
    if class_map and os.path.exists(class_map):
        with open(class_map) as f:
            class_names = json.load(f)
        num_classes = len(class_names)
    else:
        num_classes = 2  # 指定为2个类别

    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    targets, preds = [], []
    global_tp, global_fp, global_fn = 0, 0, 0
    class_stats = {0: {'tp': 0, 'fp': 0, 'fn': 0},
                   1: {'tp': 0, 'fp': 0, 'fn': 0}}

    print(f"开始验证，共 {len(image_files)} 张图像...")
    print(f"每张图像最多保留 {max_detections} 个预测框")

    # 处理每张图像
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        width, height = get_image_dimensions(img_path)
        base_name = os.path.splitext(img_file)[0]

        # 加载标注和预测
        gt_boxes = load_detection_results(
            os.path.join(gt_dir, f"{base_name}.txt"), width, height, True
        )
        pred_boxes = load_detection_results(
            os.path.join(pred_dir, f"{base_name}.txt"), width, height, False
        )

        # 转换格式
        target = convert_to_target(gt_boxes, width, height)
        prediction = convert_to_prediction(pred_boxes, width, height, max_detections)
        targets.append(target)
        preds.append(prediction)

        # 计算每张图像的TP/FP/FN
        tp, fp, fn = calculate_class_tp_fp_fn(
            target["boxes"], prediction["boxes"],
            target["labels"], prediction["labels"]
        )

        # 更新全局统计
        for cls in [0, 1]:
            class_stats[cls]['tp'] += tp.get(cls, 0)
            class_stats[cls]['fp'] += fp.get(cls, 0)
            class_stats[cls]['fn'] += fn.get(cls, 0)
            global_tp += tp.get(cls, 0)
            global_fp += fp.get(cls, 0)
            global_fn += fn.get(cls, 0)

        # 每处理10张图像打印一次进度
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(image_files)} 张图像")

    # 定义IoU阈值
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print("正在计算mAP指标...")

    # 手动计算mAP
    ap_results, map_results = calculate_map_per_class(
        preds, targets, iou_thresholds, num_classes, max_detections
    )

    # 提取关键指标
    map_50 = map_results.get('mAP_iou0.5', 0.0)

    # 计算每类AP50-95
    map_per_class = []
    for cls in range(num_classes):
        cls_aps = []
        for iou in iou_thresholds:
            ap_key = f'class{cls}_iou{iou}'
            cls_aps.append(ap_results.get(ap_key, 0.0))
        cls_map = np.mean(cls_aps) if cls_aps else 0.0
        map_per_class.append(cls_map)

    # 计算mAP50-95 - 所有类别的平均
    map_50_95 = np.mean(map_per_class) if map_per_class else 0.0

    # 提取每类AP50
    map_50_per_class = [ap_results.get(f'class{cls}_iou0.5', 0.0)
                        for cls in range(num_classes)]

    # 计算全局精确率和召回率
    class_precisions = []
    class_recalls = []

    for cls in range(num_classes):
        cls_stats = class_stats.get(cls, {'tp': 0, 'fp': 0, 'fn': 0})
        cls_precision = cls_stats['tp'] / (cls_stats['tp'] + cls_stats['fp']) if cls_stats['tp'] + cls_stats[
            'fp'] > 0 else 0.0
        cls_recall = cls_stats['tp'] / (cls_stats['tp'] + cls_stats['fn']) if cls_stats['tp'] + cls_stats[
            'fn'] > 0 else 0.0

        class_precisions.append(cls_precision)
        class_recalls.append(cls_recall)

    # 全局指标为各类别平均值
    global_precision = np.mean(class_precisions) if class_precisions else 0.0
    global_recall = np.mean(class_recalls) if class_recalls else 0.0

    # 构建结果字典
    results = {
        "all": {
            "TP": global_tp,
            "FP": global_fp,
            "FN": global_fn,
            "Precision": global_precision,
            "Recall": global_recall,
            "mAP50": map_50,
            "mAP50-95": map_50_95
        }
    }

    # 添加每类结果
    for cls in range(num_classes):
        cls_stats = class_stats.get(cls, {'tp': 0, 'fp': 0, 'fn': 0})
        cls_precision = cls_stats['tp'] / (cls_stats['tp'] + cls_stats['fp']) if cls_stats['tp'] + cls_stats[
            'fp'] > 0 else 0.0
        cls_recall = cls_stats['tp'] / (cls_stats['tp'] + cls_stats['fn']) if cls_stats['tp'] + cls_stats[
            'fn'] > 0 else 0.0

        results[f"class{cls}"] = {
            "TP": cls_stats['tp'],
            "FP": cls_stats['fp'],
            "FN": cls_stats['fn'],
            "Precision": cls_precision,
            "Recall": cls_recall,
            "mAP50": map_50_per_class[cls],
            "mAP50-95": map_per_class[cls]
        }

    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

        # 打印结果摘要
        print("\n验证结果摘要:")
        header_format = "{:<10} {:<10} {:<10} {:<10} {:<12} {:<6} {:<6} {:<6}"
        data_format = "{:<10} {:>9.3f} {:>9.3f} {:>9.3f} {:>11.3f} {:>5} {:>5} {:>5}"

        print(header_format.format('Class', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'TP', 'FP', 'FN'))
        print("-" * 85)

        # 打印全局结果
        all_row = results["all"]
        print(data_format.format(
            'all', all_row['Precision'], all_row['Recall'],
            all_row['mAP50'], all_row['mAP50-95'],
            all_row['TP'], all_row['FP'], all_row['FN']
        ))

        # 打印每个类别结果
        for cls in range(num_classes):
            cls_key = f"class{cls}"
            cls_row = results[cls_key]
            print(data_format.format(
                cls_key, cls_row['Precision'], cls_row['Recall'],
                cls_row['mAP50'], cls_row['mAP50-95'],
                cls_row['TP'], cls_row['FP'], cls_row['FN']
            ))

    print(f"\n结果已保存至: {output_file}")

    return results


def calculate_class_tp_fp_fn(gt_boxes, pred_boxes, gt_labels, pred_labels, iou_threshold=0.5):
    """计算每个类别的TP/FP/FN"""
    tp = {0: 0, 1: 0}
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}

    # 如果没有预测框或真实框，直接返回统计
    if gt_boxes.size(0) == 0:
        for label in pred_labels.tolist():
            if label in [0, 1]:
                fp[label] += 1
        return tp, fp, fn

    if pred_boxes.size(0) == 0:
        for label in gt_labels.tolist():
            if label in [0, 1]:
                fn[label] += 1
        return tp, fp, fn

    # 计算IoU矩阵
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()

    # 计算TP和FP
    for pred_idx in range(len(pred_boxes)):
        pred_label = pred_labels[pred_idx].item()
        if pred_label not in [0, 1]:  # 只处理类别0和1
            continue

        best_iou = 0.0
        best_gt_idx = -1

        # 寻找匹配的真实框
        for gt_idx in range(len(gt_boxes)):
            gt_label = gt_labels[gt_idx].item()
            if gt_label == pred_label and gt_idx not in matched_gt:
                current_iou = ious[pred_idx, gt_idx].item()
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

        if best_gt_idx != -1 and best_iou >= iou_threshold:
            tp[pred_label] += 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_label] += 1

    # 计算FN
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in matched_gt:
            gt_label = gt_labels[gt_idx].item()
            if gt_label in [0, 1]:
                fn[gt_label] += 1

    return tp, fp, fn