from PIL import ImageDraw, ImageFont, Image
import random
import numpy as np
from utils import calculate_steps
from utils import timer_decorator


@timer_decorator("绘制检测框")
def draw_detection_boxes(image, detection_boxes):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    color_map = {'seed': (255, 165, 0), 'seed_1': (0, 0, 255), 'partial': (128, 0, 128)}
    # 标签映射
    display_map = {'seed': 'healthy', 'seed_1': 'singular', 'partial': 'partial'}

    for box in detection_boxes:
        x1, y1, x2, y2, confidence, label = box
        color = color_map.get(label, (128, 128, 128))
        display_label = display_map.get(label, label)

        # 1. 绘制检测框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 2. 计算文字大小和背景框位置
        text = f"{display_label} {confidence:.2f}"
        # 获取文字相对于 (0,0) 的 bbox: (left, top, right, bottom)
        t_left, t_top, t_right, t_bottom = draw.textbbox((0, 0), text, font=font)
        text_w = t_right - t_left
        text_h = t_bottom - t_top

        # 设置 Padding（边距）
        padding = 2
        # 背景框坐标：底部紧贴 y1，左侧对齐 x1
        # [左上x, 左上y, 右下x, 右下y]
        bg_rect = [
            x1,
            y1 - text_h - (padding * 2),
            x1 + text_w + (padding * 2),
            y1
        ]

        # 3. 执行绘制
        draw.rectangle(bg_rect, fill=color)  # 绘制背景色块
        # 文字坐标：在背景框基础上偏移 padding
        draw.text((x1 + padding, y1 - text_h - padding - 1), text, fill="white", font=font)


@timer_decorator("可视化：NMS风险区检测")
def draw_nms_categorized_boxes(image, detection_data, slice_height, slice_width, step_height, step_width, margin=5):
    """
    专门为 localized_nms 准备的可视化函数：
    - 红色框：落入 get_nms_overlap_mask 判定范围，将执行 0.15 严格 NMS。
    - 蓝色框：处于安全区，执行标准 NMS 阈值。
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # 导入你刚刚在 post_processing.py 中写的新函数
    from post_processing import get_nms_overlap_mask

    # 1. 获取 NMS 专用的掩码判定结果
    # 注意：这里的 margin 必须与你 localized_nms 调用时传入的一致（比如 5 或 25）
    is_nms_overlap_mask = get_nms_overlap_mask(
        detection_data,
        slice_height,
        slice_width,
        step_height,
        step_width,
        img_w,
        img_h,
        margin
    )

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for i, box in enumerate(detection_data):
        x1, y1, x2, y2 = box[:4]
        confidence = box[4]
        label = box[5]

        is_strict = is_nms_overlap_mask[i]

        # 红色代表进入“严打区”，蓝色代表“安全区”
        color = (255, 0, 0) if is_strict else (30, 144, 255)

        # 绘制检测框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 计算重叠宽度
        ovp_w = slice_width - step_width
        ovp_h = slice_height - step_height

        # --- 1. 生成所有可能的物理边界线 (去重并排序) ---
        # 垂直线：包含每一个 step 的起点 和 每一个切片的右边界 (step + overlap)
        v_boundaries = []
        for x in range(0, img_w, step_width):
            v_boundaries.append(x)  # 重叠带起点
            v_boundaries.append(x + ovp_w)  # 重叠带终点
        v_boundaries = sorted(list(set([v for v in v_boundaries if v <= img_w])))

        # 水平线：包含每一个 step 的起点 和 每一个切片的下边界 (step + overlap)
        h_boundaries = []
        for y in range(0, img_h, step_height):
            h_boundaries.append(y)  # 重叠带起点
            h_boundaries.append(y + ovp_h)  # 重叠带终点
        h_boundaries = sorted(list(set([h for h in h_boundaries if h <= img_h])))

        # --- 2. 绘制参考线 (可选，用于调试观察) ---
        for vx in v_boundaries:
            draw.line([(vx, 0), (vx, img_h)], fill=(128, 0, 128, 60), width=1)
        for hy in h_boundaries:
            draw.line([(0, hy), (img_w, hy)], fill=(128, 0, 128, 60), width=1)

        # 绘制标签背景和文字
        zone_tag = "STRICT(0.15)" if is_strict else "NORMAL"
        text = f"{zone_tag} {label} {confidence:.2f}"

        # 为了美观，把文字背景框也涂色
        text_bbox = draw.textbbox((x1, y1 - 18), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - (text_bbox[3] - text_bbox[1]) - 2), text, fill="white", font=font)

    return image


@timer_decorator("可视化：全边界贴合检测")
def draw_categorized_boxes(image, detection_data, slice_height, slice_width, step_height, step_width, margin=10):
    """
    完善版本：考虑了重叠带的左右/上下双边界
    - 红色框：触碰了任何切片物理缝隙或图像边缘。
    - 黄色粗线：具体贴合的那条物理边界。
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # 计算重叠宽度
    ovp_w = slice_width - step_width
    ovp_h = slice_height - step_height

    # --- 1. 生成所有可能的物理边界线 (去重并排序) ---
    # 垂直线：包含每一个 step 的起点 和 每一个切片的右边界 (step + overlap)
    v_boundaries = []
    for x in range(0, img_w, step_width):
        v_boundaries.append(x)  # 重叠带起点
        v_boundaries.append(x + ovp_w)  # 重叠带终点
    v_boundaries = sorted(list(set([v for v in v_boundaries if v <= img_w])))

    # 水平线：包含每一个 step 的起点 和 每一个切片的下边界 (step + overlap)
    h_boundaries = []
    for y in range(0, img_h, step_height):
        h_boundaries.append(y)  # 重叠带起点
        h_boundaries.append(y + ovp_h)  # 重叠带终点
    h_boundaries = sorted(list(set([h for h in h_boundaries if h <= img_h])))

    # --- 2. 绘制参考线 (可选，用于调试观察) ---
    for vx in v_boundaries:
        draw.line([(vx, 0), (vx, img_h)], fill=(0, 0, 255), width=1) # fill=(128, 0, 128, 60)
    for hy in h_boundaries:
        draw.line([(0, hy), (img_w, hy)], fill=(0, 0, 255, 60), width=1)

    # --- 3. 遍历检测框进行判定 ---
    for box in detection_data:
        x1, y1, x2, y2 = box[:4]

        is_touching = False
        active_v_lines = []
        active_h_lines = []

        # 检查垂直边界贴合 (左右边)
        for vx in v_boundaries:
            if np.abs(x1 - vx) <= margin:
                is_touching = True
                active_v_lines.append((vx, y1, vx, y2))
            if np.abs(x2 - vx) <= margin:
                is_touching = True
                active_v_lines.append((vx, y1, vx, y2))

        # 检查水平边界贴合 (上下边)
        for vy in h_boundaries:
            if np.abs(y1 - vy) <= margin:
                is_touching = True
                active_h_lines.append((x1, vy, x2, vy))
            if np.abs(y2 - vy) <= margin:
                is_touching = True
                active_h_lines.append((x1, vy, x2, vy))

        # --- 4. 绘图执行 ---
        # color = (255, 0, 0) if is_touching else (30, 144, 255)
        color = (255, 0, 0) if is_touching else (255, 165, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return image
"""
        if is_touching:
            # 绘制黄色高亮边
            for line_coords in active_v_lines:
                draw.line(line_coords, fill=(255, 255, 0), width=6)
            for line_coords in active_h_lines:
                draw.line(line_coords, fill=(255, 255, 0), width=6)


    return image
"""

"""
@timer_decorator("绘制区域分类检测框")
def draw_categorized_boxes(image, detection_data, slice_height, slice_width, step_height, step_width, margin=5):
    
    # 可视化调试函数：同步更新参数以支持新的重叠带判定逻辑
    
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # 导入更新后的判定逻辑
    from post_processing import get_overlap_zone_mask

    # 确保传入 get_overlap_zone_mask 的参数与 post_processing.py 中的定义完全一致
    is_overlap_mask = get_overlap_zone_mask(
        detection_data,
        slice_height,
        slice_width,
        step_height,
        step_width,
        img_w,
        img_h,
        margin
    )

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for i, box in enumerate(detection_data):
        x1, y1, x2, y2, confidence, label = box[:6]

        is_overlap = is_overlap_mask[i]
        color = (255, 0, 0) if is_overlap else (0, 0, 255)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        zone_type = "OVERLAP" if is_overlap else "SAFE"
        text = f"[{zone_type}] {label}"

        text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - (text_bbox[3] - text_bbox[1])), text, fill="white", font=font)

    return image
"""


@timer_decorator("绘制切片边界")
def draw_slice_borders(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, font_size,
                       text_opacity, overlap_height, overlap_width):
    draw = ImageDraw.Draw(image)

    step_height, step_width = calculate_steps(slice_height, slice_width,
                                              overlap_height_ratio, overlap_width_ratio,
                                              overlap_height, overlap_width)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    row_number = 1
    top = 0
    while top < image.height:
        bottom = min(top + slice_height, image.height)
        if bottom - top <= 5:
            top += step_height
            continue

        column_number = 1
        left = 0
        while left < image.width:
            right = min(left + slice_width, image.width)
            if right - left <= 5:
                left += step_width
                continue

            draw.rectangle([left, top, right, bottom], outline="blue", width=1)
            # text = f"({row_number},{column_number})"
            # text_position = (left + 5, top + 5)
            # text_color = (255, 0, 0, int(text_opacity))
            # draw.text(text_position, text, font=font, fill=text_color)

            column_number += 1
            left += step_width
            if right >= image.width:
                break

        row_number += 1
        top += step_height
        if bottom >= image.height:
            break


def save_image(image, output_path):
    """保存图像 - 使用TurboJPEG加速"""
    image.save(output_path, "jpeg", method=1)