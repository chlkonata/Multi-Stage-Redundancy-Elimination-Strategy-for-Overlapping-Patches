from estimate import compute_iou, compute_giou
import numpy as np
from utils import timer_decorator, Timer  # 添加导入


def validate_bbox(box):
    """验证边界框的有效性
    if len(box) < 5:
        return False
    if box[0] >= box[2] or box[1] >= box[3]:
        return False
    if box[4] < 0 or box[4] > 1:
        return False"""
    return True


def is_center_in_box(center_box, target_box):
    """
    判断一个框的中心是否在另一个框内部

    参数:
        center_box: 待判断中心的框 [x1, y1, x2, y2, ...]
        target_box: 目标框 [x1, y1, x2, y2, ...]

    返回:
        bool: 中心点是否在目标框内
    """
    # 计算中心点坐标
    center_x = (center_box[0] + center_box[2]) / 2
    center_y = (center_box[1] + center_box[3]) / 2

    # 判断中心点是否在目标框内
    return (target_box[0] <= center_x <= target_box[2] and
            target_box[1] <= center_y <= target_box[3])


def is_point_in_box(point, box):
    """
    判断点是否在框内
    """
    x, y = point
    return (box[0] <= x <= box[2] and box[1] <= y <= box[3])


def get_nms_overlap_mask(detection_data, slice_height, slice_width, step_height, step_width, img_w, img_h, margin=5):
    """
    专门为 localized_nms 设计的原始版本：
    判定标准：只要检测框的任一边界距离切片物理缝隙或图像边缘 <= margin，即判定为处于重叠/边缘区。
    """
    if len(detection_data) == 0:
        return np.array([], dtype=bool)

    # 1. 计算切片重叠区域的物理坐标线
    ovp_w = slice_width - step_width
    ovp_h = slice_height - step_height

    v_lines = []
    for x in range(0, img_w, step_width):
        v_lines.append(x)          # 切片左边界
        v_lines.append(x + ovp_w)  # 切片右边界
    v_lines = np.array(sorted(list(set([v for v in v_lines if v <= img_w]))))

    h_lines = []
    for y in range(0, img_h, step_height):
        h_lines.append(y)          # 切片上边界
        h_lines.append(y + ovp_h)  # 切片下边界
    h_lines = np.array(sorted(list(set([h for h in h_lines if h <= img_h]))))

    # 2. 提取框坐标
    boxes = np.array(detection_data)[:, :4].astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 3. 判定边缘距离（向量化计算）
    # 只要左、右、上、下任一维度贴近物理分割线，即为 True
    near_v1 = np.any(np.abs(x1[:, None] - v_lines) <= margin, axis=1)
    near_v2 = np.any(np.abs(x2[:, None] - v_lines) <= margin, axis=1)
    near_h1 = np.any(np.abs(y1[:, None] - h_lines) <= margin, axis=1)
    near_h2 = np.any(np.abs(y2[:, None] - h_lines) <= margin, axis=1)

    return near_v1 | near_v2 | near_h1 | near_h2


def get_overlap_zone_mask(detection_data, slice_height, slice_width, step_height, step_width, img_w, img_h, margin=5):
    """
    严格贴合判定：只有当检测框的边【撞】在切片物理边界上时才返回 True。
    """
    if len(detection_data) == 0:
        return np.array([], dtype=bool)

    # 1. 计算所有物理切割线 (这些是切片的起点和终点，不含中间重叠区)
    v_lines = []
    for x in range(0, img_w, step_width):
        v_lines.append(x)  # 切片左边缘线
        # 计算该切片的右边缘线
        right_line = x + slice_width
        if right_line < img_w:
            v_lines.append(right_line)
    v_lines = np.array(sorted(list(set(v_lines))))

    h_lines = []
    for y in range(0, img_h, step_height):
        h_lines.append(y)  # 切片上边缘线
        bottom_line = y + slice_height
        if bottom_line < img_h:
            h_lines.append(bottom_line)
    h_lines = np.array(sorted(list(set(h_lines))))

    # 2. 提取所有框的四边
    boxes = np.array(detection_data)[:, :4].astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 3. 严格判定：是否有一条边与切割线重合
    # np.abs(...) <= margin 判定是否“撞墙”
    touch_v1 = np.any(np.abs(x1[:, None] - v_lines) <= margin, axis=1)
    touch_v2 = np.any(np.abs(x2[:, None] - v_lines) <= margin, axis=1)
    touch_h1 = np.any(np.abs(y1[:, None] - h_lines) <= margin, axis=1)
    touch_h2 = np.any(np.abs(y2[:, None] - h_lines) <= margin, axis=1)

    # 只要任意一条边贴合了物理切割线，就是真正的 Touching
    return touch_v1 | touch_v2 | touch_h1 | touch_h2


def remove_highly_aligned_overlaps(detection_data, is_touching_mask, alignment_threshold=3, iou_threshold=0.85):
    """
    功能：去除高度对齐冗余框，并进行置信度继承。
    满足条件时：保留面积大的（完整框），移除面积小的（不完整框），并将高置信度赋予保留框。
    """
    if len(detection_data) < 2:
        return detection_data

    # 将 tuple 转换为 list 方便修改置信度 [x1, y1, x2, y2, conf, label]
    data_list = [list(box) for box in detection_data]
    boxes = np.array([box[:4] for box in data_list])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    n = len(boxes)

    # 记录需要删除的索引
    to_remove = set()

    for i in range(n):
        for j in range(i + 1, n):
            # 基础条件：至少一个框在边缘区 (is_touching_mask 为 True)
            if not (is_touching_mask[i] or is_touching_mask[j]):
                continue

            # 条件 2: IoU 判定 (建议此处传入 0.85 左右)
            iou = compute_iou(boxes[i], boxes[j])
            if iou < iou_threshold:
                continue

            # 条件 3: 边对齐判定 (计算四条边的绝对差值)
            diffs = np.abs(boxes[i] - boxes[j])
            aligned_sides = np.sum(diffs <= alignment_threshold)

            if aligned_sides >= 3:
                # 确定保留者(keep)和被删除者(rm)
                if areas[i] >= areas[j]:
                    keep_idx, rm_idx = i, j
                else:
                    keep_idx, rm_idx = j, i

                # --- 置信度赋值逻辑 ---
                # 将两个框中较高的置信度赋予给保留下来的那个“完整框”
                # 这样即使完整框原本分低，现在也拿到了高分，能在 NMS 中幸存
                # highest_conf = max(data_list[keep_idx][4], data_list[rm_idx][4])
                # data_list[keep_idx][4] = highest_conf

                # 标记删除较小的框
                to_remove.add(rm_idx)

    # 最终根据索引过滤掉冗余框，并转回 tuple 格式
    return [tuple(data_list[k]) for k in range(n) if k not in to_remove]


def normal_nms(detection_data, iou_threshold, slice_height, slice_width, step_height, step_width, image_width,
                  image_height, margin=5):
    """
    普通版 NMS 包装器：
    接口与原 localized_nms 保持完全一致，但内部忽略切片/边缘逻辑，
    直接调用标准全局 NMS 进行处理。
    """
    if not detection_data:
        return []

    # 直接调用你代码中已有的标准 nms 函数
    # 注意：标准 nms 内部已经包含了 @timer_decorator 和置信度排序逻辑
    return nms(detection_data, iou_threshold=iou_threshold)


def localized_nms(detection_data, iou_threshold, slice_height, slice_width, step_height, step_width, image_width,
                  image_height, margin=5):
    """
    方案 A：取消物理隔离
    所有框参与全局 NMS 排序，但处于重叠带的框在判定时执行更严苛的 0.15 阈值。
    """
    if not detection_data:
        return []

    # 1. 获取掩码，标记哪些框属于重叠带
    is_overlap_mask = get_nms_overlap_mask(
        detection_data, slice_height, slice_width, step_height, step_width, image_width, image_height, margin
    )

    # 2. 调用修改后的增强型 nms
    # 我们把 is_overlap_mask 作为一个属性带进 nms
    return nms_adaptive_threshold(detection_data, is_overlap_mask, iou_threshold)


@timer_decorator("自适应NMS处理", default_verbose=True)
def nms_adaptive_threshold(detection_boxes, is_overlap_mask, default_iou_threshold=0.5):
    if len(detection_boxes) == 0:
        return []

    # 1. 转换为 numpy 数组进行快速计算
    boxes_array = np.array([(box[0], box[1], box[2], box[3], box[4]) for box in detection_boxes])
    labels = np.array([box[5] for box in detection_boxes])
    # 将 mask 也转为 numpy 数组并跟随排序
    overlap_mask = np.array(is_overlap_mask)

    # 2. 按置信度排序
    order = boxes_array[:, 4].argsort()[::-1]
    boxes_array = boxes_array[order]
    labels = labels[order]
    overlap_mask = overlap_mask[order]

    keep_boxes = []

    while boxes_array.shape[0] > 0:
        # 取当前最高分框
        current_box = boxes_array[0]
        current_label = labels[0]
        current_is_overlap = overlap_mask[0]

        keep_boxes.append((
            current_box[0], current_box[1], current_box[2], current_box[3],
            current_box[4], current_label
        ))

        if boxes_array.shape[0] == 1:
            break

        # 3. 计算与剩余框的 IOU
        ious = compute_batch_iou(boxes_array[0:1], boxes_array[1:])
        remaining_labels = labels[1:]
        remaining_overlap = overlap_mask[1:]

        # 4. 【核心逻辑：动态阈值判定】
        # 如果当前框或者被比较的框中，有一个处于重叠区，就用 0.15；否则用默认阈值。
        # 只要有一方在“严打区”，阈值就收紧。
        dynamic_thresholds = np.where(
            (current_is_overlap | remaining_overlap),
            0.5,
            default_iou_threshold
        )

        # 判定保留掩码：
        # 保留条件：(IOU < 动态阈值) OR (标签不同)
        mask = (ious < dynamic_thresholds) | (remaining_labels != current_label)

        boxes_array = boxes_array[1:][mask]
        labels = remaining_labels[mask]
        overlap_mask = remaining_overlap[mask]

    return keep_boxes


@timer_decorator("NMS处理", default_verbose=True)
def nms(detection_boxes, iou_threshold=0.5, verbose=False):
    if len(detection_boxes) == 0:
        return []

    # 1. 转换为numpy数组
    boxes_array = np.array([(box[0], box[1], box[2], box[3], box[4]) for box in detection_boxes])
    labels = np.array([box[5] for box in detection_boxes])  # 转换为 numpy array 方便切片

    # 2. 按置信度排序
    order = boxes_array[:, 4].argsort()[::-1]
    boxes_array = boxes_array[order]
    labels = labels[order]

    keep_boxes = []

    while boxes_array.shape[0] > 0:
        # 取置信度最高的框
        current_box = boxes_array[0]
        current_label = labels[0]
        keep_boxes.append((
            current_box[0], current_box[1], current_box[2], current_box[3],
            current_box[4], current_label
        ))

        if boxes_array.shape[0] == 1:
            break

        # 3. 计算与剩余框的 IOU
        ious = compute_batch_iou(boxes_array[0:1], boxes_array[1:])

        # 4. 【核心修改】：类别感知逻辑
        # 只有当 (IOU > 阈值) 且 (标签相同) 时，才标记为抑制（False）
        # 反之，(IOU < 阈值) 或者 (标签不同) 的框都要保留（True）
        remaining_labels = labels[1:]

        # mask = True 表示保留，False 表示剔除
        # 逻辑：如果不重叠 OR 类别不一样，就保留
        mask = (ious < iou_threshold) | (remaining_labels != current_label)

        boxes_array = boxes_array[1:][mask]
        labels = remaining_labels[mask]

    return keep_boxes

"""
def merge_truncated_overlaps_v2(detection_boxes, is_touching_mask, alignment_threshold=5, confidence_transfer=True):
    """"""
    仅针对 touching 场景处理特定重叠：保留完整框并吸收截断框的置信度。

    参数:
        detection_boxes: 原始检测框列表
        is_touching_mask: get_overlap_zone_mask 生成的掩码，标记框是否触及物理缝隙
        alignment_threshold: 判定边对齐的像素阈值，默认为 5 像素
        confidence_transfer: 是否将截断高分赋予完整低分框
    """"""
    if len(detection_boxes) < 2:
        return detection_boxes

    # 转换为 list 方便修改
    boxes = [list(box) for box in detection_boxes]
    n = len(boxes)
    to_remove = set()

    for i in range(n):
        # 约束条件 1: 当前框已标记删除，或当前框不属于 touching 场景，则跳过
        if i in to_remove or not is_touching_mask[i]:
            continue

        for j in range(n):
            if i == j or j in to_remove:
                continue

            # 只要有一方处于 touching 状态，即视为可能存在截断重叠的风险区
            # (如果 j 也不满足 touching，且 i 也不满足，上面的 if 已经过滤了)

            box1, box2 = boxes[i], boxes[j]

            # 计算四边差距
            diffs = np.abs(np.array(box1[:4]) - np.array(box2[:4]))

            # 判定对齐特征：至少有两条边高度重合 (例如左、右、上三边重合)
            aligned_sides = diffs <= alignment_threshold
            if np.sum(aligned_sides) >= 2:

                # 确定完整框(面积大)与截断框(面积小)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                complete_idx, truncated_idx = (i, j) if area1 > area2 else (j, i)

                # 计算 IoU 确保是同一目标
                iou = compute_iou(boxes[complete_idx][:4], boxes[truncated_idx][:4])

                if iou > 0.7:
                    # 核心逻辑：转移高置信度给完整框
                    if confidence_transfer:
                        boxes[complete_idx][4] = max(boxes[complete_idx][4], boxes[truncated_idx][4])

                    to_remove.add(truncated_idx)
                    break

    return [tuple(box) for idx, box in enumerate(boxes) if idx not in to_remove]
"""


def compute_batch_iou(box, boxes):
    """批量计算IOU - 向量化实现"""
    # 计算交集
    x1 = np.maximum(box[0, 0], boxes[:, 0])
    y1 = np.maximum(box[0, 1], boxes[:, 1])
    x2 = np.minimum(box[0, 2], boxes[:, 2])
    y2 = np.minimum(box[0, 3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # 计算面积
    area_box = (box[0, 2] - box[0, 0] + 1) * (box[0, 3] - box[0, 1] + 1)
    area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # 计算IOU
    union = area_box + area_boxes - intersection
    iou = intersection / union

    return iou


"""
@timer_decorator("检测框合并-调试版", default_verbose=True)
def merge_detection_boxes_debug(
    detection_boxes,
    merge_iou_threshold=0.8,
    min_centers_in_box=2,
    center_overlap=True,
    verbose=False
):
    if len(detection_boxes) == 0:
        return []

    # ---------- Step 0: 验证边界框 ----------
    valid_boxes = [box for box in detection_boxes if validate_bbox(box)]
    if not valid_boxes:
        return []

    n = len(valid_boxes)

    # ---------- Step 1: 计算中心点 ----------
    centers = []
    for box in valid_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append((cx, cy))

    # ---------- Step 2: 初始 special 判定 ----------
    initial_special = set()

    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue
            if is_point_in_box(centers[j], valid_boxes[i]):
                count += 1

        if count >= min_centers_in_box:
            initial_special.add(i)
            if verbose:
                print(f"框{i}包含{count}个中心点，初始标记为 special")

    # ---------- Step 3: special 结构级扩散（关键修复） ----------
    special_boxes = set(initial_special)
    changed = True

    while changed:
        changed = False
        for i in range(n):
            if i in special_boxes:
                continue
            for j in special_boxes:
                if is_point_in_box(centers[i], valid_boxes[j]) or \
                   is_point_in_box(centers[j], valid_boxes[i]):
                    special_boxes.add(i)
                    changed = True
                    if verbose:
                        print(f"框{i} 与 special 框{j} 中心点关联，扩散为 special")
                    break

    # ---------- Step 4: BFS 合并普通框 ----------
    merged_boxes = []
    processed_indices = set()

    for i in range(n):
        if i in processed_indices:
            continue

        # special 结构：完全不合并
        if i in special_boxes:
            merged_boxes.append(valid_boxes[i])
            processed_indices.add(i)
            if verbose:
                print(f"框{i} 属于 special 结构，直接保留")
            continue

        # BFS 合并普通框
        to_merge = set([i])
        queue = [i]
        processed_indices.add(i)

        while queue:
            k = queue.pop(0)
            box_k = valid_boxes[k]
            center_k = centers[k]

            if not center_overlap:
                continue

            for j in range(n):
                if j in processed_indices or j in special_boxes:
                    continue

                box_j = valid_boxes[j]
                center_j = centers[j]

                if is_point_in_box(center_k, box_j) or is_point_in_box(center_j, box_k):
                    to_merge.add(j)
                    processed_indices.add(j)
                    queue.append(j)

                    if verbose:
                        print(f"框{k} 与 框{j} 满足中心点包含，加入合并队列")

        # ---------- Step 5: 执行合并 ----------
        if len(to_merge) > 1:
            merged_box = merge_box_group([valid_boxes[idx] for idx in to_merge])
            merged_boxes.append(merged_box)
            if verbose:
                print(f"合并 {len(to_merge)} 个普通框")
        else:
            merged_boxes.append(valid_boxes[i])

    return merged_boxes
"""


@timer_decorator("检测框去重-保留高分版", default_verbose=True)
def merge_detection_boxes_debug(
        detection_boxes,
        is_touching_mask,
        buffer=1,
        verbose=False
):
    if len(detection_boxes) == 0:
        return []

    # 1. 预处理：保留有效框，并附带原始索引以方便追踪
    indexed_boxes = []
    for i, box in enumerate(detection_boxes):
        if validate_bbox(box):
            # 将原始索引、框体、是否贴边 封装在一起
            indexed_boxes.append({
                'box': box,
                'is_touching': is_touching_mask[i],
                'conf': box[4],
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            })

    if not indexed_boxes:
        return []

    # 2. 按置信度从高到低排序 (关键：高分的先选，作为保留基准)
    indexed_boxes.sort(key=lambda x: x['conf'], reverse=True)

    keep_boxes = []
    eliminated_indices = set()

    def is_point_in_box_robust(point, box, b):
        x, y = point
        return (box[0] - b <= x <= box[2] + b and
                box[1] - b <= y <= box[3] + b)

    for i in range(len(indexed_boxes)):
        if i in eliminated_indices:
            continue

        # 当前框是目前最高分的，肯定保留
        current_item = indexed_boxes[i]
        keep_boxes.append(current_item['box'])

        # 3. 检查剩余的框，如果有冲突则剔除
        for j in range(i + 1, len(indexed_boxes)):
            if j in eliminated_indices:
                continue

            compare_item = indexed_boxes[j]

            # 判定条件：(i或j贴边) AND (中心点包含)
            if current_item['is_touching'] or compare_item['is_touching']:
                if is_point_in_box_robust(current_item['center'], compare_item['box'], buffer) or \
                        is_point_in_box_robust(compare_item['center'], current_item['box'], buffer):

                    eliminated_indices.add(j)
                    if verbose:
                        print(f"检测到重叠: 舍弃低分框 (conf:{compare_item['conf']}), "
                              f"保留高分框 (conf:{current_item['conf']})")

    return keep_boxes

"""
@timer_decorator("检测框合并-全触发版", default_verbose=True)
def merge_detection_boxes_debug(
        detection_boxes,
        is_touching_mask,
        buffer=1,
        verbose=False
):
    if len(detection_boxes) == 0:
        return []

    valid_boxes = [box for box in detection_boxes if validate_bbox(box)]
    if not valid_boxes:
        return []

    n = len(valid_boxes)
    # 预计算中心点
    centers = []
    for box in valid_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append((cx, cy))

    merged_boxes = []
    processed_indices = set()

    def is_point_in_box_robust(point, box, b):
        x, y = point
        return (box[0] - b <= x <= box[2] + b and
                box[1] - b <= y <= box[3] + b)

    for i in range(n):
        if i in processed_indices:
            continue

        # 1. 寻找所有与框 i 满足合并条件的伙伴
        # 只要 (i 或 j 贴边) 且 (中心点包含)，就加入合并簇
        to_merge_indices = [i]

        for j in range(n):
            if i == j or j in processed_indices:
                continue

            # 核心修改：准入门槛判定
            # 只要 i 或 j 任何一个是 Touching，就有合并的必要
            if is_touching_mask[i] or is_touching_mask[j]:
                # 几何包含判定
                if is_point_in_box_robust(centers[i], valid_boxes[j], buffer) or \
                        is_point_in_box_robust(centers[j], valid_boxes[i], buffer):
                    to_merge_indices.append(j)

        # 2. 根据搜寻结果处理
        if len(to_merge_indices) > 1:
            # 形成合并簇，将簇内所有索引标为已处理
            merged_box = merge_box_group([valid_boxes[idx] for idx in to_merge_indices])
            merged_boxes.append(merged_box)
            for idx in to_merge_indices:
                processed_indices.add(idx)
            if verbose:
                print(f"触发集合合并: 索引 {to_merge_indices} 合并为一个新框")
        else:
            # 没有找到任何合并对象，框 i 独立存在
            merged_boxes.append(valid_boxes[i])
            processed_indices.add(i)

    return merged_boxes


def merge_box_group(boxes):
    # 合并一组框为一个框
    if not boxes:
        return None

    # 计算并集
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    # 置信度：取最高
    confidence = max(box[4] for box in boxes)

    # 标签赋予规则：
    # 1. 若两个标签中存在partial与其他值（seed, seed_1），则合并后改为另一个标签值
    # 2. 若不存在partial，则优先赋予高置信度的标签

    # 收集所有标签
    labels = [box[5] for box in boxes]

    # 检查是否有partial标签
    has_partial = 'partial' in labels

    if has_partial:
        # 找出所有非partial标签
        non_partial_labels = [label for label in labels if label != 'partial']

        if non_partial_labels:
            # 存在非partial标签，选择其中一个
            # 优先选择最高置信度的非partial标签
            max_conf = -1
            selected_label = None

            for box in boxes:
                if box[5] != 'partial' and box[4] > max_conf:
                    max_conf = box[4]
                    selected_label = box[5]

            # 如果找到了非partial标签
            if selected_label:
                label = selected_label
            else:
                # 理论上不会走到这里，但为了安全
                # 从非partial标签中取第一个
                label = non_partial_labels[0]
        else:
            # 只有partial标签，保持partial
            label = 'partial'
    else:
        # 没有partial标签，选择最高置信度的标签
        max_conf = -1
        selected_label = None

        for box in boxes:
            if box[4] > max_conf:
                max_conf = box[4]
                selected_label = box[5]

        label = selected_label if selected_label else labels[0]

    return (x1, y1, x2, y2, confidence, label)

"""


def clean_partial_labels(detection_data, lambda_h=0.75, alpha=0.8):
    """
    实现图中 Sf 的边缘标签清洗操作：剔除与完整框高度重合且置信度较低的 partial 框。

    参数:
        detection_data: 原始检测框列表，格式为 [(x1, y1, x2, y2, score, label), ...]
        lambda_h: IoU 阈值，默认为 0.85
        alpha: 置信度衰减系数，默认为 0.8
    返回:
        Sc: 清洗后的检测框列表
    """
    if not detection_data:
        return []

    # 1. 记 Sf 为所有检测框集合
    # 2. 记 L(Sf) 为检测框标签为 'partial' 的集合
    partial_boxes = [b for b in detection_data if b[5] == 'partial']
    # 记非 partial 的框为对比基准
    normal_boxes = [b for b in detection_data if b[5] != 'partial']

    if not partial_boxes or not normal_boxes:
        return detection_data

    # 存储需要剔除的 partial 框的 ID (使用 id() 避免列表元素重复导致的判断错误)
    to_remove_ids = set()

    for b_normal in normal_boxes:
        score_b = b_normal[4]

        for b_partial in partial_boxes:
            # 如果该 partial 框已经被标记为剔除，跳过
            if id(b_partial) in to_remove_ids:
                continue

            # 计算 IoU (使用文件中已有的 compute_iou)
            iou_val = compute_iou(b_normal[:4], b_partial[:4])

            # 判定条件 P(b):
            # 1. IoU(b, b') > lambda_h (0.85)
            # 2. alpha * Db > Db' (0.8 * 完整框分数 > 碎片框分数)
            if iou_val > lambda_h:
                if (alpha * score_b) > b_partial[4]:
                    to_remove_ids.add(id(b_partial))

    # 3. 执行 Sc = Sf \ P(b)
    # 保留不在剔除集合中的所有框
    sc = [b for b in detection_data if id(b) not in to_remove_ids]

    return sc
