from PIL import Image, ImageOps
import os
import datetime
import time
from functools import wraps
import multiprocessing
import torch
import functools


# 支持verbose控制的计时器装饰器
def timer_decorator(description="", default_verbose=True):
    """
    可控制详细输出的计时器装饰器
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从 kwargs 中获取 verbose 参数，如果没有则使用装饰器默认值
            func_verbose = kwargs.get('verbose', default_verbose)

            # 如果kwargs中没有，尝试从args中根据参数名获取
            if 'verbose' not in kwargs:
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_verbose = bound_args.arguments.get('verbose', default_verbose)

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            if func_verbose:
                print(f"[计时器] {description}: {elapsed_time:.4f}秒")

            return result

        return wrapper

    return decorator


# 为了向后兼容，保留原来的 Timer 类
class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0
        self.children_time = 0  # 记录子计时器的时间

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        net_time = self.elapsed_time - self.children_time
        print(f"[计时器] {self.name}: 总{self.elapsed_time:.4f}秒, 净{net_time:.4f}秒")

    def add_children_time(self, time):
        """添加子计时器的时间"""
        self.children_time += time

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


# 支持verbose的Timer类
class VerboseTimer:
    """可控制详细输出的计时器类"""

    def __init__(self, name="", verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed_time = 0
        self.children_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        net_time = self.elapsed_time - self.children_time

        if self.verbose:
            print(f"[计时器] {self.name}: 总{self.elapsed_time:.4f}秒, 净{net_time:.4f}秒")

    def add_children_time(self, time):
        """添加子计时器的时间"""
        self.children_time += time

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


# 全局计时器管理器
class TimerManager:
    def __init__(self):
        self.timers = {}

    def start_timer(self, name):
        self.timers[name] = time.time()

    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            print(f"[计时器] {name}: {elapsed:.4f}秒")
            del self.timers[name]
            return elapsed
        return 0


# 创建全局计时器实例
timer_manager = TimerManager()


def create_output_directory(output_dir):
    """
    确保输出目录存在
    """
    os.makedirs(output_dir, exist_ok=True)


def get_image_dimensions(image_path):
    """
    获取图像高度和宽度
    """
    image = Image.open(image_path)
    return image.size, image


def calculate_steps(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, overlap_height,
                    overlap_width):
    """
    根据切片高度宽度和重叠比例或重叠像素值计算步长
    """
    # 如果指定了重叠像素值，优先使用
    if overlap_height is not None:
        step_height = slice_height - overlap_height
    else:
        step_height = int(slice_height * (1 - overlap_height_ratio))

    if overlap_width is not None:
        step_width = slice_width - overlap_width
    else:
        step_width = int(slice_width * (1 - overlap_width_ratio))

    return step_height, step_width


def crop_image(image, slice_box, slice_width, slice_height):
    """
    裁剪图像并处理超出边界的情况
    """
    sliced_img = image.crop(slice_box)
    if slice_box[3] > image.height or slice_box[2] > image.width:
        sliced_img = ImageOps.pad(sliced_img, (slice_width, slice_height), color=(128, 128, 128))
    return sliced_img


def save_settings(output_dir, settings):
    """
    保存参数设置到TXT文件
    """
    settings_path = os.path.join(output_dir, "settings.txt")
    with open(settings_path, "w") as f:
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")


def get_current_time_formatted():
    """
    获取当前时间并格式化为字符串
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def get_file_name(image_path):
    """
    提取不带扩展名的文件名
    """
    file_name, _ = os.path.splitext(os.path.basename(image_path))
    return file_name


def generate_output_directory(image_path):
    """
    根据图像路径和当前时间生成输出目录路径
    """
    formatted_time = get_current_time_formatted()
    file_name = get_file_name(image_path)
    output_dir = f"./slice/slices_{file_name}_{formatted_time}"
    return output_dir


def generate_multi_output_directory(base_name="predict"):
    """
    根据当前时间生成唯一的输出目录
    """
    formatted_time = get_current_time_formatted()
    output_dir = f"./slice/{base_name}_{formatted_time}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_image_or_directory(image_path):
    """
    检查传入的路径是否是文件夹或图像文件，并返回图像文件列表。

    如果是图像文件，返回该文件的列表；
    如果是文件夹，遍历文件夹返回其中所有图像文件的列表。

    支持的图像格式：.png, .jpg, .jpeg, .bmp, .tiff
    """
    # 判断是否为文件夹
    if os.path.isdir(image_path):
        # 遍历文件夹中的图像文件
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        return image_files
    elif os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 如果是单个图像文件，直接返回该文件的列表
        return [image_path]
    else:
        # 返回空列表，如果既不是文件夹也不是图像文件
        print(f"Invalid image path or unsupported file type: {image_path}")
        return []


def save_detection_results(output_dir, image_path, detection_boxes, img_w, img_h):
    """
    保存检测结果为YOLO格式的txt文件
    """
    # 创建结果目录
    results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(results_dir, exist_ok=True)

    # 生成结果文件名
    file_name = get_file_name(image_path)
    result_path = os.path.join(results_dir, f"{file_name}.txt")

    # 转换并保存结果
    with open(result_path, "w") as f:
        for box in detection_boxes:
            # box格式: [x1, y1, x2, y2, conf, class_id]
            x_center = (box[0] + box[2]) / (2 * img_w)
            y_center = (box[1] + box[3]) / (2 * img_h)
            width = (box[2] - box[0]) / img_w
            height = (box[3] - box[1]) / img_h

            # YOLO格式: class_id conf x_center y_center width height
            f.write(f"{int(box[5])} {box[4]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_detection_results_direct(output_dir, image_path, detection_boxes, img_w, img_h, name_to_id):
    """
    直接保存检测结果 - 优化数据流转版本
    避免中间格式转换
    """
    results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(results_dir, exist_ok=True)

    file_name = get_file_name(image_path)
    result_path = os.path.join(results_dir, f"{file_name}.txt")

    with open(result_path, "w") as f:
        for box in detection_boxes:
            # 直接从检测数据中提取信息
            x1, y1, x2, y2, conf, label = box
            class_id = name_to_id.get(label, -1)

            if class_id != -1:
                # 直接计算YOLO格式
                x_center = (x1 + x2) / (2 * img_w)
                y_center = (y1 + y2) / (2 * img_h)
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                # 一次性写入，减少I/O操作
                f.write(f"{class_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_detection_results_batch(output_dir, image_paths, all_detection_boxes, img_sizes, name_to_id):
    """
    批量保存检测结果 - 减少文件I/O开销
    """
    results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(results_dir, exist_ok=True)

    # 批量处理所有图片的结果
    for i, image_path in enumerate(image_paths):
        detection_boxes = all_detection_boxes[i]
        img_w, img_h = img_sizes[i]
        file_name = get_file_name(image_path)
        result_path = os.path.join(results_dir, f"{file_name}.txt")

        with open(result_path, "w") as f:
            for box in detection_boxes:
                x1, y1, x2, y2, conf, label = box
                class_id = name_to_id.get(label, -1)

                if class_id != -1:
                    x_center = (x1 + x2) / (2 * img_w)
                    y_center = (y1 + y2) / (2 * img_h)
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    f.write(f"{class_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def calculate_optimal_parallelism():
    """计算最优并行度"""
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if gpu_count > 0:
        # GPU模式：每个GPU分配2-4个进程
        return min(gpu_count * 4, cpu_count // 2, 4)
    else:
        # CPU模式：基于核心数
        return min(cpu_count - 1, 8)


# 为了向后兼容，创建一个简化的verbose_timer函数
def verbose_timer(name, verbose=True):
    """创建可控制详细输出的计时器"""
    return VerboseTimer(name, verbose)