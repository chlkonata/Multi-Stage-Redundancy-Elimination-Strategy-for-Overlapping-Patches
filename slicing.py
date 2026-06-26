from utils import *
from detection import detect_objects, detect_full_image
from post_processing import (nms, merge_detection_boxes_debug, get_overlap_zone_mask,
                             localized_nms, remove_highly_aligned_overlaps, clean_partial_labels, normal_nms)
from draw import draw_detection_boxes, save_image, draw_slice_borders, draw_categorized_boxes, draw_nms_categorized_boxes
from ultralytics import YOLO
import os
import time
import asyncio
from tqdm import tqdm
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import gc
import torch
import threading
import pickle
import tempfile

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# 全局模型变量
_global_model = None

# 使用多进程安全的 Manager 来共享计时数据
_manager = None
_shared_timers = None

# 全局详细输出标志
_verbose_output = True


def init_shared_timers():
    """初始化共享计时器"""
    global _manager, _shared_timers
    if _manager is None:
        _manager = multiprocessing.Manager()
        _shared_timers = _manager.dict({
            'model_load': 0.0,
            'slice': 0.0,
            'prediction': 0.0,
            'postprocess': 0.0,
            'export_files': 0.0
        })


def create_model_cache(model_path):
    """创建模型配置缓存"""
    print("主进程预加载模型中...")
    start_time = time.time()

    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 只序列化配置信息，不序列化模型本身
    model_config = {
        'model_path': model_path,
        'names': model.names,
        'device': device,
        'overrides': {
            'verbose': False,
            'device': device
        }
    }

    # 创建临时缓存文件
    temp_dir = tempfile.mkdtemp()
    cache_file = os.path.join(temp_dir, 'model_config.pkl')

    with open(cache_file, 'wb') as f:
        pickle.dump(model_config, f)

    load_time = time.time() - start_time
    print(f"模型配置缓存创建完成，耗时: {load_time:.2f}秒")

    # 立即释放主进程中的模型
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache_file


def init_worker_with_cache(cache_file):
    """使用缓存配置初始化工作进程"""
    global _global_model

    # print(f"工作进程 {multiprocessing.current_process().name} 初始化中...")

    # 从缓存加载模型配置
    with open(cache_file, 'rb') as f:
        model_config = pickle.load(f)

    # 每个工作进程独立加载模型，但使用相同的配置
    model_path = model_config['model_path']
    device = model_config['device']

    _global_model = YOLO(model_path)
    # 不需要设置 names，因为模型加载时会自动设置
    _global_model.overrides.update(model_config['overrides'])

    # print(f"工作进程 {multiprocessing.current_process().name} 初始化完成")


def get_worker_model():
    """获取工作进程中的模型"""
    return _global_model


def cleanup_model_cache(cache_file):
    """清理模型缓存文件"""
    try:
        if cache_file and os.path.exists(cache_file):
            cache_dir = os.path.dirname(cache_file)
            os.remove(cache_file)
            os.rmdir(cache_dir)
            print("模型缓存文件已清理")
    except Exception as e:
        print(f"清理缓存文件时出错: {e}")


class ModelManager:
    """模型管理器 - 优化模型调用方式"""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_path, device='auto'):
        """获取模型实例，支持单例模式"""
        with cls._lock:
            if model_path not in cls._instances:
                print(f"加载模型: {model_path}")
                cls._instances[model_path] = YOLO(model_path)

                # 优化模型配置
                model = cls._instances[model_path]
                model.overrides['verbose'] = False
                model.overrides['device'] = device

            return cls._instances[model_path]

    @classmethod
    def clear_models(cls):
        """清理所有模型实例"""
        with cls._lock:
            for model in cls._instances.values():
                if hasattr(model, 'model'):
                    del model.model
                del model
            cls._instances.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def update_shared_timer(timer_name, duration):
    """更新共享计时器"""
    global _shared_timers
    if _shared_timers and timer_name in _shared_timers:
        _shared_timers[timer_name] += duration


def print_final_timings(shared_timers):
    """打印最终的计时统计，格式与 predict.py 一致"""
    print("\n=== 计时统计 ===")
    if shared_timers['model_load'] > 0:
        print(f"Model loaded in {shared_timers['model_load']:.6f} seconds.")
    if shared_timers['slice'] > 0:
        print(f"Slicing performed in {shared_timers['slice']:.6f} seconds.")
    if shared_timers['prediction'] > 0:
        print(f"Prediction performed in {shared_timers['prediction']:.6f} seconds.")


def reset_shared_timers():
    """重置共享计时器"""
    global _shared_timers
    if _shared_timers:
        for key in _shared_timers.keys():
            _shared_timers[key] = 0.0


class VerboseTimer:
    """可控制详细输出的计时器类"""

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed_time = 0.0
        self.children_times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        if self.verbose:
            print(f"[计时器] {self.name}: 总{self.elapsed_time:.4f}秒")

    def add_children_time(self, children_time):
        """添加子计时器时间"""
        self.children_times.append(children_time)


def verbose_timer(name, verbose=True):
    """创建可控制详细输出的计时器"""
    return VerboseTimer(name, verbose)


def process_single_image(img_file, class_names, name_to_id, output_dir,
                         slice_height, slice_width, overlap_height_ratio,
                         overlap_width_ratio, overlap_height, overlap_width,
                         confidence_threshold, yolo_iou, yolo_conf, max_det,
                         full_size_detect, imgsz, use_nms, iou_threshold,
                         draw_borders, font_size, text_opacity, merge_boxes,
                         merge_iou_threshold, save_detection,
                         save_image_result=True,
                         verbose=True,
                         show_progress=True):
    """处理单张图像 - 使用工作进程模型版本"""

    # 为每个进程创建本地计时器
    local_timers = {
        'slice': 0.0,
        'prediction': 0.0,
        'postprocess': 0.0,
        'export_files': 0.0
    }

    with verbose_timer(f"处理单张图像 {os.path.basename(img_file)}", verbose) as total_timer:
        # 使用工作进程中的模型
        worker_model = get_worker_model()

        file_name = os.path.basename(img_file)

        with verbose_timer("获取图像尺寸", verbose):
            (image_width, image_height), image = get_image_dimensions(img_file)

        create_output_directory(output_dir)

        with verbose_timer("计算步长", verbose):
            step_height, step_width = calculate_steps(slice_height, slice_width,
                                                      overlap_height_ratio, overlap_width_ratio,
                                                      overlap_height, overlap_width)

        detection_data = []

        # 切片计时开始 - 与 predict.py 一致：只包含切片位置计算
        slice_start_time = time.time()

        # 预计算切片位置
        with verbose_timer("预计算切片位置", verbose):
            slices = []
            top = 0
            while top < image_height:
                bottom = min(top + slice_height, image_height)
                if bottom - top > 5:
                    left = 0
                    while left < image_width:
                        right = min(left + slice_width, image_width)
                        if right - left > 5:
                            slices.append((left, top, right, bottom))
                        left += step_width
                        if right >= image.width:
                            break
                top += step_height
                if bottom >= image_height:
                    break

        # 切片计时结束 - 只包含预计算，不包含图像裁剪
        slice_duration = time.time() - slice_start_time
        local_timers['slice'] = slice_duration

        # ---------------------------批量处理切片-----------------------------------------
        batch_size = 1
        num_slices = len(slices)

        # 预测计时开始 - 与 predict.py 一致：包含图像裁剪 + 模型推理 + 后处理
        prediction_start_time = time.time()

        # 图像裁剪 - 计入预测时间（因为这是推理的必要准备）
        crop_total_time = 0.0
        with verbose_timer("所有批次总时间", verbose) as batch_total_timer:
            for i in range(0, num_slices, batch_size):
                batch_slices = slices[i:i + batch_size]
                sliced_images = []
                offsets = []

                # 图像裁剪计时
                crop_start_time = time.time()
                with verbose_timer(f"批次 {i // batch_size + 1} 图像裁剪", verbose):
                    for idx, (left, top, right, bottom) in enumerate(batch_slices):
                        slice_box = (left, top, right, bottom)
                        sliced_img = crop_image(image, slice_box, right - left, bottom - top)
                        sliced_images.append(sliced_img)
                        offsets.append((left, top))
                crop_duration = time.time() - crop_start_time
                crop_total_time += crop_duration

                # 目标检测 - 计入预测时间
                detection_start_time = time.time()
                with verbose_timer(f"批次 {i // batch_size + 1} 目标检测", verbose):
                    batch_boxes = detect_objects(
                        worker_model,
                        sliced_images,
                        confidence_threshold,
                        offsets,
                        yolo_iou,
                        yolo_conf,
                        max_det,
                        verbose=verbose,
                    )
                detection_duration = time.time() - detection_start_time
                local_timers['prediction'] += detection_duration

                # ==================== 修正后：保存切片检测结果与绘图 ====================
                if save_image_result or save_detection:
                    # 为当前原图创建专门的切片保存目录
                    slices_save_dir = os.path.join(output_dir, "slices", os.path.splitext(file_name)[0])
                    os.makedirs(slices_save_dir, exist_ok=True)

                    for idx, (left, top, right, bottom) in enumerate(batch_slices):
                        current_slice_boxes_local = []
                        for box in batch_boxes:
                            box_cx = (box[0] + box[2]) / 2
                            box_cy = (box[1] + box[3]) / 2

                            if left <= box_cx < right and top <= box_cy < bottom:
                                local_box = list(box)
                                local_box[0] -= left
                                local_box[1] -= top
                                local_box[2] -= left
                                local_box[3] -= top
                                current_slice_boxes_local.append(local_box)

                        slice_base_name = f"slice_{i + idx}_{left}_{top}"

                        # 1. 绘制并保存切片图像
                        if save_image_result and len(current_slice_boxes_local) > 0:
                            slice_img_draw = sliced_images[idx].copy()
                            draw_detection_boxes(slice_img_draw, current_slice_boxes_local)
                            save_image(slice_img_draw, os.path.join(slices_save_dir, f"{slice_base_name}.jpg"))

                        # 2. 保存切片检测结果文本 (修正了类型转换问题)
                        if save_detection and len(current_slice_boxes_local) > 0:
                            slice_txt_path = os.path.join(slices_save_dir, f"{slice_base_name}.txt")
                            sw = right - left
                            sh = bottom - top
                            with open(slice_txt_path, "w") as f:
                                for lb in current_slice_boxes_local:
                                    # --- 关键修正：安全获取类别 ID ---
                                    raw_cls = lb[5]
                                    # 如果是字符串则从字典查，否则直接转 int
                                    cls_id = name_to_id.get(raw_cls, raw_cls) if isinstance(raw_cls, str) else raw_cls

                                    x_center = (lb[0] + lb[2]) / (2 * sw)
                                    y_center = (lb[1] + lb[3]) / (2 * sh)
                                    w_norm = (lb[2] - lb[0]) / sw
                                    h_norm = (lb[3] - lb[1]) / sh
                                    f.write(
                                        f"{int(cls_id)} {lb[4]:.6f} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                # ==================================================================

                detection_data.extend(batch_boxes)

                # 添加批次内存清理计时
                with verbose_timer(f"批次 {i // batch_size + 1} 内存清理", verbose):
                    del sliced_images, batch_boxes
                    if i % (batch_size * 4) == 0:
                        gc.collect()

        # 将图像裁剪时间计入预测时间
        local_timers['prediction'] += crop_total_time

        # 记录批次总时间到父计时器
        if hasattr(batch_total_timer, 'add_children_time'):
            batch_total_timer.add_children_time(batch_total_timer.elapsed_time)

        # 添加最终内存清理计时
        with verbose_timer("批次处理后的最终内存清理", verbose):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 全图推理 - 计入预测时间
        if full_size_detect:
            full_size_start_time = time.time()
            with verbose_timer("全图检测", verbose):
                full_size_boxes = detect_full_image(
                    worker_model, img_file, confidence_threshold, imgsz, yolo_iou, yolo_conf, max_det,
                    verbose=verbose
                )
                detection_data.extend(full_size_boxes)
            full_size_duration = time.time() - full_size_start_time
            local_timers['prediction'] += full_size_duration

        # 在执行任何过滤和抑制前，保留一份原始数据的副本用于类别找回
        raw_detection_data_copy = list(detection_data)

        # 后处理流程 - 计入预测时间（与 predict.py 一致）
        postprocess_start_time = time.time()

        if detection_data:
            """"""
            # --------融合策略之一-------
            # 1. 先计算一次 Mask
            current_mask = get_overlap_zone_mask(
                detection_data, slice_height, slice_width,
                step_height, step_width, image_width, image_height, margin=5
            )

            # 2. 【新增步骤】在 NMS 之前拦截并处理对齐框
            with verbose_timer("预处理：对齐截断框过滤", verbose):
                detection_data = remove_highly_aligned_overlaps(
                    detection_data,
                    current_mask,
                    alignment_threshold=3,
                    iou_threshold=0.65
                )
            # --------融合策略之一-------

            # 2. NMS 处理 (修复后的调用)
            if use_nms and detection_data:

                with verbose_timer("局部NMS处理 (仅重叠区)", verbose):
                    # 传入所有必需的 8 个位置参数
                    detection_data = localized_nms(
                    # detection_data = normal_nms(
                        detection_data,
                        iou_threshold,
                        slice_height,  # 新增
                        slice_width,  # 新增
                        step_height,
                        step_width,
                        image_width,
                        image_height,
                        margin=5
                    )
                """

                #--- 实验组 A: 常规 NMS ---
                with verbose_timer("实验 A: 常规 NMS 处理", verbose):
                    # 注意：nms 函数只接受 3 个参数 (detection_boxes, iou_threshold, verbose)
                    detection_data = nms(
                        detection_data,
                        iou_threshold=iou_threshold,
                        verbose=verbose
                    )
"""
            # --- 4.新增步骤：类别修复 ---
            if detection_data:
                with verbose_timer("边缘标签清洗", verbose):
                    # 仅保留标签不是 'partial' 的检测框
                    detection_data = [box for box in detection_data if box[5] != 'partial']


            # 3. 检测框合并
            if merge_boxes and detection_data:
                with verbose_timer("检测框合并", verbose):
                    # detection_data = merge_detection_boxes(detection_data, slice_height, slice_width, overlap_height
                                                           # , overlap_width, (image_width, image_height),merge_iou_threshold)
                    # 1. 重新计算针对当前检测结果的 mask（因为 NMS 后数量变了）
                    current_mask = get_overlap_zone_mask(
                        detection_data,
                        slice_height, slice_width,
                        step_height, step_width,
                        image_width, image_height,
                        margin=3
                    )

                    # 2. 传入正确的 mask 进行合并
                    detection_data = merge_detection_boxes_debug(
                        detection_data,
                        is_touching_mask=current_mask,
                        verbose=False
                    )
            # detection_data = [box for box in detection_data if box[5] != 'partial']
            # --- 4.新增步骤：类别修复 ---
            """
            if detection_data:
                with verbose_timer("边缘标签清洗", verbose):
                    detection_data = clean_partial_labels(
                        detection_data,
                        lambda_h=0.85,
                        alpha=0.8
                    )
            """

        postprocess_duration = time.time() - postprocess_start_time
        local_timers['prediction'] += postprocess_duration

        # 预测计时结束 - 包含图像裁剪 + 模型推理 + 后处理
        total_prediction_duration = time.time() - prediction_start_time
        # 确保预测时间与总时间一致（包含所有操作）
        local_timers['prediction'] = total_prediction_duration

        # 记录后处理总时间到父计时器
        if hasattr(total_timer, 'add_children_time'):
            total_timer.add_children_time(total_prediction_duration)

        # 保存和绘制 - 添加总保存计时器
        export_start_time = time.time()
        with verbose_timer("保存和绘制总时间", verbose) as save_draw_timer:
            if save_detection and detection_data:
                with verbose_timer("保存检测结果", verbose):
                    save_detection_results_direct(
                        output_dir, img_file, detection_data, image_width, image_height, name_to_id
                    )

            # 只有在需要保存图片时才进行绘制操作
            if save_image_result:
                with verbose_timer("绘制检测框", verbose):
                    draw_detection_boxes(image, detection_data)

                # --- 调试用的分类图调用修复 ---
                if verbose:
                    with verbose_timer("绘制区域诊断图", verbose):
                        base_name = os.path.splitext(file_name)[0]

                        # 1. 绘制原有的融合判定诊断图 (红色代表 Touching，用于 Merge 逻辑)
                        # 注意：这里的 margin 应该对应你 merge 逻辑使用的数值（如 25）
                        debug_merge_image = draw_categorized_boxes(
                            image.copy(),  # 使用 copy 防止绘图叠加
                            detection_data,
                            slice_height,
                            slice_width,
                            step_height,
                            step_width,
                            margin=5
                        )
                        save_image(debug_merge_image, os.path.join(output_dir, f"debug_merge_{base_name}.jpg"))

                        # 2. 增加：绘制 NMS 风险区诊断图 (专门针对 localized_nms 的 0.15 严打区)
                        # 注意：这里的 margin 必须与你 localized_nms 内部使用的 margin 完全一致
                        debug_nms_image = draw_nms_categorized_boxes(
                            image.copy(),
                            detection_data,
                            slice_height,
                            slice_width,
                            step_height,
                            step_width,
                            margin=5  # 这里设为 5，因为这是你最初方案的默认值
                        )
                        save_image(debug_nms_image, os.path.join(output_dir, f"debug_nms_{base_name}.jpg"))

                # --------------------------

                if draw_borders:
                    with verbose_timer("绘制切片边界", verbose):
                        draw_slice_borders(image,
                                           slice_height,
                                           slice_width,
                                           overlap_height_ratio,
                                           overlap_width_ratio,
                                           font_size,
                                           text_opacity,
                                           overlap_height,
                                           overlap_width)

                with verbose_timer("保存结果图像", verbose):
                    base_name = os.path.splitext(file_name)[0]
                    save_image(image, os.path.join(output_dir, f"{base_name}.jpg"))
            else:
                # 如果不保存图片，跳过所有绘制操作
                if verbose:
                    print(f"跳过绘制和保存图片: {file_name}")

        export_duration = time.time() - export_start_time
        local_timers['export_files'] += export_duration

        # 记录保存绘制总时间到父计时器
        if hasattr(total_timer, 'add_children_time'):
            total_timer.add_children_time(save_draw_timer.elapsed_time)

        # 添加其他系统开销计时
        with verbose_timer("其他系统开销", verbose):
            pass

        # 返回文件名和本地计时数据
        return file_name, local_timers


def process_single_image_wrapper(args):
    """包装函数用于多进程池处理单张图像"""
    (
        img_file, class_names, name_to_id, output_dir,
        slice_height, slice_width, overlap_height_ratio, overlap_width_ratio,
        overlap_height, overlap_width,
        confidence_threshold, yolo_iou, yolo_conf, max_det, full_size_detect,
        imgsz, use_nms, iou_threshold, draw_borders, font_size, text_opacity,
        merge_boxes, merge_iou_threshold, save_detection,
        save_image_result,
        verbose
    ) = args

    # 添加进程初始化计时
    with verbose_timer("进程初始化和事件循环创建", verbose):
        pass

    try:
        result = process_single_image(
            img_file, class_names, name_to_id, output_dir,
            slice_height, slice_width, overlap_height_ratio,
            overlap_width_ratio, overlap_height, overlap_width,
            confidence_threshold, yolo_iou,
            yolo_conf, max_det, full_size_detect, imgsz, use_nms,
            iou_threshold, draw_borders, font_size, text_opacity,
            merge_boxes, merge_iou_threshold, save_detection,
            save_image_result,
            verbose,
            show_progress=False
        )

        # 返回结果和计时数据
        if isinstance(result, tuple) and len(result) == 2:
            file_name, local_timers = result
            return f"完成: {file_name}", local_timers
        else:
            return f"完成: {result}", {}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}", {}
    finally:
        with verbose_timer("进程清理和资源释放", verbose):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def slice_image(image_path,
                slice_height,
                slice_width,
                overlap_height_ratio,
                overlap_width_ratio,
                overlap_height=None,
                overlap_width=None,
                model_path=None,
                confidence_threshold=None,
                yolo_iou=0.7,
                yolo_conf=0.25,
                max_det=300,
                use_nms=False,
                full_size_detect=False,
                imgsz=800,
                iou_threshold=0.5,
                draw_borders=False,
                font_size=12,
                text_opacity=255,
                merge_boxes=False,
                merge_iou_threshold=0.5,
                save_detection=False,
                save_image_result=True,
                verbose=True):
    """将图像切成多个小片段 - 使用模型配置缓存版本"""

    # 设置全局详细输出标志
    global _verbose_output
    _verbose_output = verbose

    # 初始化共享计时器
    init_shared_timers()
    reset_shared_timers()

    # 在函数开始添加重叠参数验证
    if overlap_height is not None and overlap_height >= slice_height:
        raise ValueError(f"重叠高度 {overlap_height} 不能大于等于切片高度 {slice_height}")
    if overlap_width is not None and overlap_width >= slice_width:
        raise ValueError(f"重叠宽度 {overlap_width} 不能大于等于切片宽度 {slice_width}")

    with verbose_timer("图像路径处理", verbose):
        image_files = process_image_or_directory(image_path)

    if not image_files:
        print(f"No valid images found in {image_path}")
        return

    # 生成输出目录
    if len(image_files) > 1:
        output_dir = generate_multi_output_directory("predict")
    else:
        output_dir = generate_output_directory(os.path.basename(image_files[0]))

    # 创建结果目录
    detection_results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(detection_results_dir, exist_ok=True)

    # 预加载模型配置并创建缓存
    if verbose:
        print("创建模型配置缓存...")
    model_load_start_time = time.time()
    with verbose_timer("模型配置缓存创建", verbose):
        cache_file = create_model_cache(model_path)

        # 临时加载模型获取类别信息（立即释放）
        temp_model = YOLO(model_path)
        class_names = temp_model.names
        name_to_id = {v: k for k, v in class_names.items()}
        del temp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_load_duration = time.time() - model_load_start_time
    update_shared_timer('model_load', model_load_duration)

    print(f"检测类别: {list(class_names.values())}")

    # 打印使用的重叠参数
    if overlap_height is not None or overlap_width is not None:
        print(f"使用像素重叠: 高度={overlap_height}, 宽度={overlap_width}")
    else:
        print(f"使用比例重叠: 高度比例={overlap_height_ratio}, 宽度比例={overlap_width_ratio}")

    # 打印参数设置
    print(f"保存检测图片: {'是' if save_image_result else '否'}")
    print(f"详细输出: {'是' if verbose else '否'}")

    # 记录检测开始时间
    detection_start_time = time.time()
    total_images = len(image_files)

    # 准备多进程参数
    with verbose_timer("准备多进程参数", verbose):
        worker_args = []
        for img_file in image_files:
            worker_args.append((
                img_file, class_names, name_to_id, output_dir,
                slice_height, slice_width, overlap_height_ratio, overlap_width_ratio,
                overlap_height, overlap_width,
                confidence_threshold, yolo_iou, yolo_conf, max_det, full_size_detect,
                imgsz, use_nms, iou_threshold, draw_borders, font_size, text_opacity,
                merge_boxes, merge_iou_threshold, save_detection,
                save_image_result,
                verbose
            ))

    # 智能计算进程数
    max_workers = calculate_optimal_parallelism()
    print(f"使用 {max_workers} 个工作进程 (CPU: {multiprocessing.cpu_count()}, GPU: {torch.cuda.device_count()})")

    print(f"处理 {total_images} 张图像")
    pbar = tqdm(total=total_images, desc="总进度")

    completed = 0
    errors = 0
    success_count = 0

    # 用于收集所有工作进程的计时数据
    all_local_timers = []

    try:
        with verbose_timer("多进程处理", verbose):
            with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=init_worker_with_cache,
                    initargs=(cache_file,)
            ) as executor:
                # 提交任务
                futures = [executor.submit(process_single_image_wrapper, args) for args in worker_args]

                # 处理结果
                for future in futures:
                    try:
                        result, local_timers = future.result()
                        if result.startswith("ERROR:"):
                            print(f"\n处理失败: {result}")
                            errors += 1
                        else:
                            success_count += 1
                            pbar.set_description(result)
                            # 收集计时数据
                            if local_timers:
                                all_local_timers.append(local_timers)
                        completed += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n处理失败: {e}")
                        errors += 1
                        pbar.update(1)

    except Exception as e:
        print(f"多进程处理出错: {e}")
        errors = total_images
    finally:
        pbar.close()

        # 清理缓存文件
        cleanup_model_cache(cache_file)

    # 汇总所有工作进程的计时数据
    if all_local_timers:
        for local_timers in all_local_timers:
            for key in local_timers:
                if key in _shared_timers:
                    _shared_timers[key] += local_timers[key]

    # 清理模型管理器
    ModelManager.clear_models()

    # 清理全局模型变量
    global _global_model
    if _global_model is not None:
        del _global_model
        _global_model = None

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 性能统计
    total_detection_time = time.time() - detection_start_time
    total_time = time.time() - detection_start_time
    fps = success_count / total_detection_time if total_detection_time > 0 else float('inf')

    print(f"\n=== 处理完成 ===")
    print(f"总图像数量: {total_images}")
    print(f"成功处理: {success_count}, 失败: {errors}")
    print(f"检测时间: {total_detection_time:.2f} 秒")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"FPS: {fps:.2f} (成功图像数量/检测时间)")
    print(f"平均每张图像: {total_detection_time / success_count if success_count > 0 else 0:.2f} 秒")

    # 打印与 predict.py 格式一致的计时统计
    print_final_timings(dict(_shared_timers))

    # 保存参数设置
    settings = {
        "image_path": image_path,
        "slice_height": slice_height,
        "slice_width": slice_width,
        "overlap_height_ratio": overlap_height_ratio,
        "overlap_width_ratio": overlap_width_ratio,
        "overlap_height": overlap_height,
        "overlap_width": overlap_width,
        "output_dir": output_dir,
        "model": model_path,
        "max_det": max_det,
        "confidence_threshold": confidence_threshold,
        "yolo_iou": yolo_iou,
        "yolo_conf": yolo_conf,
        "use_nms": use_nms,
        "full_size_detect": full_size_detect,
        "imgsz": imgsz,
        "iou_threshold": iou_threshold,
        "draw_borders": draw_borders,
        "font_size": font_size,
        "text_opacity": text_opacity,
        "merge_boxes": merge_boxes,
        "merge_iou_threshold": merge_iou_threshold,
        "save_detection": save_detection,
        "save_image_result": save_image_result,
        "verbose": verbose,
        "detection_time": total_detection_time,
        "total_time": total_time,
        "fps": fps,
        "total_images": total_images,
        "workers": max_workers,
        "success": success_count,
        "errors": errors,
        "timings": dict(_shared_timers)
    }

    save_settings(output_dir, settings)

    return output_dir


# 保存检测结果函数保持不变
def save_detection_results(output_dir, image_path, detection_boxes, img_w, img_h):
    """保存检测结果为YOLO格式的txt文件"""
    results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(results_dir, exist_ok=True)

    file_name = get_file_name(image_path)
    result_path = os.path.join(results_dir, f"{file_name}.txt")

    with open(result_path, "w") as f:
        for box in detection_boxes:
            x_center = (box[0] + box[2]) / (2 * img_w)
            y_center = (box[1] + box[3]) / (2 * img_h)
            width = (box[2] - box[0]) / img_w
            height = (box[3] - box[1]) / img_h
            f.write(f"{int(box[5])} {box[4]} {x_center:.17f} {y_center:.17f} {width:.17f} {height:.17f}\n")