import time  # 导入时间模块
from utils import Timer  # 添加导入
from slicing import slice_image
from utils import generate_output_directory

# 图像路径(图像或文件夹均可)
# image_path = "./image/train/seed_64.jpg"
# image_path = "./image/train/seed_55.jpg"
# image_path = "./image/train/wx_420.jpg"
# image_path = "./image/test/zxa_325.jpg"
# image_path = "./image/train/"
# image_path = "./image/seed(6).jpg"
# image_path = "./image/test/zxa_212.jpg"
# image_path = "./image/0206_data/020602.jpg"
# image_path = "./image/0206_data/"
image_path = "./label/img/"
# image_path = "./label/img_all/"

# 切片高宽
slice_height = 800
slice_width = 800

# 切片高宽重叠比例
overlap_height_ratio = 0 # 0.5
overlap_width_ratio = 0  # 0.5

# 或者直接指定像素值（优先级更高）(指定像素值时，忽视重叠比例)
overlap_height = 150
overlap_width = 150

# 生成输出目录
output_dir = generate_output_directory(image_path)

# 加载YOLOv8模型
model_path = './models/last_YOLOV8n+WIoU v3(tal-ciou)+EMA6 6：4 0910 new.pt'
# model_path = './models/last_YOLOV8n+WIoU v3(tal-ciou)+EMA6 6：4 nonpartial new.pt'
# model_path = './models/best.pt'
# model_path = './models/yolov8n4060-152.pt'
# model_path = './models/yolo11n-1000.pt'

if __name__ == '__main__':
    # 使用Timer类进行更精确的计时
    with Timer("整个程序运行") as total_timer:
        # 记录开始时间
        start_time = time.time()

        # 执行切片操作
        output_dir = slice_image(image_path,
                                 slice_height,
                                 slice_width,
                                 overlap_height_ratio,
                                 overlap_width_ratio,
                                 overlap_height,
                                 overlap_width,
                                 model_path,
                                 max_det=3000,
                                 yolo_iou=0.7,
                                 yolo_conf=0.25,
                                 confidence_threshold=0.5,
                                 use_nms=True,
                                 full_size_detect=False,
                                 imgsz=[4000, 3008],
                                 iou_threshold=0.7,  # 0.7-0.15
                                 draw_borders=True,
                                 font_size=36,
                                 text_opacity=255,
                                 merge_boxes=True,
                                 save_detection=True,
                                 save_image_result=False,
                                 verbose=False)

        # 计算并打印运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== 总体统计 ===")
        print(f"预测完成! 总耗时: {elapsed_time:.2f}秒")