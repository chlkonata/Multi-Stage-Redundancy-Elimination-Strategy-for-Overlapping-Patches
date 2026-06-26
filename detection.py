import os
import torch
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
from utils import timer_decorator, Timer  # 使用更新后的timer_decorator

# 创建线程池执行器（全局共享）
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


@timer_decorator("YOLO检测结果处理", default_verbose=True)
def process_yolo_detections_batch(results, confidence_threshold, offset_list, verbose=False):
    """高效处理批量YOLO检测结果 - 优化数据流转版本"""
    detection_boxes = []

    if not isinstance(results, list):
        results = [results]
        offset_list = [offset_list]

    for i, res in enumerate(results):
        left, top = offset_list[i]
        boxes = res.boxes

        if boxes is not None and len(boxes) > 0:
            # 使用向量化操作一次性处理所有框
            conf_mask = boxes.conf.flatten() >= confidence_threshold

            if conf_mask.any():
                # 一次性获取所有有效数据
                valid_boxes = boxes.xyxy[conf_mask].cpu().numpy()
                conf_values = boxes.conf[conf_mask].cpu().numpy()
                cls_values = boxes.cls[conf_mask].cpu().numpy().astype(int)

                # 批量应用偏移量
                valid_boxes[:, 0] += left  # x1
                valid_boxes[:, 1] += top  # y1
                valid_boxes[:, 2] += left  # x2
                valid_boxes[:, 3] += top  # y2

                # 批量获取标签名称
                labels = [res.names[cls] for cls in cls_values]

                # 一次性构建最终数据结构
                for j in range(len(valid_boxes)):
                    detection_boxes.append((
                        valid_boxes[j, 0],  # x1
                        valid_boxes[j, 1],  # y1
                        valid_boxes[j, 2],  # x2
                        valid_boxes[j, 3],  # y2
                        float(conf_values[j]),  # confidence
                        labels[j]  # label
                    ))

    return detection_boxes


@timer_decorator("YOLO批量检测", default_verbose=True)
def detect_objects(model, input_data, confidence_threshold, offset_list, iou=0.7, conf=0.25, max_det=300, verbose=True):
    """使用YOLO模型进行批量对象检测 - 优化调用版本"""
    # 统一输入格式处理
    if not isinstance(input_data, list):
        input_data = [input_data]
        offset_list = [offset_list]

    # 优化推理参数 - 固定配置，减少参数解析开销
    results = model.predict(
        source=input_data,
        verbose=False,           # 关闭详细输出
        iou=iou,                # 固定IOU阈值
        conf=conf,              # 固定置信度阈值
        max_det=max_det,        # 固定最大检测数
        batch=len(input_data),  # 自适应批大小
        device=model.overrides.get('device', 'cpu'),  # 使用模型预设设备
        augment=False,          # 关闭数据增强
        agnostic_nms=False,     # 关闭agnostic NMS
        half=False,             # 关闭半精度（提高兼容性）
        dnn=False,              # 关闭DNN后端
        # 移除不必要的参数，减少配置开销
    )

    return process_yolo_detections_batch(results, confidence_threshold, offset_list, verbose=verbose)


async def async_detect_objects(model, input_data, confidence_threshold, offset_list,
                                       iou=0.7, conf=0.25, max_det=300, verbose=False):
    """异步版本的对象检测函数 - 优化调用"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: detect_objects(model, input_data, confidence_threshold, offset_list, iou, conf, max_det, verbose)
    )


@timer_decorator("YOLO全图检测", default_verbose=True)
def detect_full_image(model, image_path, confidence_threshold,
                                imgsz=None, iou=0.7, conf=0.25, max_det=300, verbose=False):
    """使用YOLOv8模型直接检测整个图片 - 优化调用版本"""
    # 优化全图检测参数
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        verbose=False,
        iou=iou,
        conf=conf,
        max_det=max_det,
        device=model.overrides.get('device', 'cpu'),
        augment=False,
        half=False
    )

    detection_boxes = []
    for result in results:
        if result.boxes is None:
            continue

        # 使用向量化操作处理检测结果
        conf_mask = result.boxes.conf.flatten() >= confidence_threshold

        if conf_mask.any():
            valid_boxes = result.boxes.xyxy[conf_mask].cpu().numpy()
            conf_values = result.boxes.conf[conf_mask].cpu().numpy()
            cls_values = result.boxes.cls[conf_mask].cpu().numpy().astype(int)
            labels = [result.names[cls] for cls in cls_values]

            for i, box in enumerate(valid_boxes):
                x1, y1, x2, y2 = box.tolist()
                detection_boxes.append((x1, y1, x2, y2, float(conf_values[i]), labels[i]))

    return detection_boxes


async def async_detect_full_image(model, image_path, confidence_threshold,
                                          imgsz=None, iou=0.7, conf=0.25, max_det=300, verbose=True):
    """异步版本的全图检测函数 - 优化调用"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: detect_full_image(model, image_path, confidence_threshold, imgsz, iou, conf, max_det, verbose)
    )


# 确保优化函数可以被导入
__all__ = [
    'process_yolo_detections_batch',
    'detect_objects',
    'async_detect_objects',
    'detect_full_image',
    'async_detect_full_image'
]