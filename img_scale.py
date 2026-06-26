import os
from PIL import Image


def resize_image(input_path, output_path=None, max_size=800, quality=85):
    """
    将图片长边压缩到指定像素，短边等比例缩放

    参数:
    input_path: 输入图片路径
    output_path: 输出图片路径（如果为None，则在原文件名后添加_resized）
    max_size: 最大边长（默认800像素）
    quality: 输出图片质量（1-100，默认85）
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 获取原始尺寸
            original_width, original_height = img.size

            # 计算缩放比例
            if original_width >= original_height:
                # 横图或正方形，以宽为基准
                new_width = max_size
                new_height = int(original_height * (max_size / original_width))
            else:
                # 竖图，以高为基准
                new_height = max_size
                new_width = int(original_width * (max_size / original_height))

            # 调整图片尺寸
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 确定输出路径
            if output_path is None:
                name, ext = os.path.splitext(input_path)
                output_path = f"{name}_resized{ext}"

            # 保存图片
            if img.mode in ('RGBA', 'LA', 'P'):
                # 处理透明背景的图片
                resized_img.save(output_path, quality=quality, optimize=True)
            else:
                # 普通图片
                resized_img.save(output_path, quality=quality, optimize=True)

            print(f"✓ 图片已成功调整尺寸: {os.path.basename(input_path)}")
            print(f"  原始尺寸: {original_width} x {original_height}")
            print(f"  新尺寸: {new_width} x {new_height}")
            print(f"  输出路径: {output_path}")

            return True

    except Exception as e:
        print(f"✗ 处理图片时出错 {os.path.basename(input_path)}: {e}")
        return False


def batch_resize_images(folder_path, output_folder=None, max_size=800, quality=85):
    """
    批量处理文件夹中的所有图片

    参数:
    folder_path: 文件夹路径
    output_folder: 输出文件夹路径（如果为None，则在原文件夹内保存）
    max_size: 最大边长
    quality: 输出图片质量
    """
    # 支持的图片格式
    supported_formats = (
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP')

    if not os.path.exists(folder_path):
        print(f"✗ 文件夹不存在: {folder_path}")
        return

    # 创建输出文件夹（如果指定了输出文件夹）
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✓ 创建输出文件夹: {output_folder}")

    success_count = 0
    total_count = 0

    print(f"\n开始处理文件夹: {folder_path}")
    print("=" * 60)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            total_count += 1
            input_path = os.path.join(folder_path, filename)

            # 设置输出路径
            if output_folder:
                output_path = os.path.join(output_folder, filename)
            else:
                output_path = None

            if resize_image(input_path, output_path, max_size, quality):
                success_count += 1

    print("=" * 60)
    print(f"批量处理完成: {success_count}/{total_count} 个文件处理成功")

    return success_count, total_count


def main():
    """
    主函数 - 在这里设置参数
    """
    # ==================== 参数设置区域 ====================

    # 输入文件夹路径
    INPUT_FOLDER = r"C:\Users\Administrator\Desktop\SSI\label\img"  # 替换为您的图片文件夹路径

    # 输出文件夹路径（设为None则在原文件夹保存，文件名后加_resized）
    OUTPUT_FOLDER = r"C:\Users\Administrator\Desktop\SSI\label\800img"  # 例如: r"C:\Users\YourUsername\Pictures\resized"

    # 最大边长（像素）
    MAX_SIZE = 800

    # 图片质量 (1-100)
    QUALITY = 85

    # ==================== 参数设置结束 ====================

    print("图片批量尺寸调整工具")
    print("=" * 50)
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出位置: {OUTPUT_FOLDER if OUTPUT_FOLDER else '原文件夹（添加_resized后缀）'}")
    print(f"最大边长: {MAX_SIZE} 像素")
    print(f"图片质量: {QUALITY}")
    print("=" * 50)

    # 确认开始处理
    confirm = input("是否开始处理? (y/n): ").strip().lower()
    if confirm not in ('y', 'yes', '是'):
        print("操作已取消")
        return

    # 执行批量处理
    success, total = batch_resize_images(
        folder_path=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        max_size=MAX_SIZE,
        quality=QUALITY
    )

    if success > 0:
        print(f"\n🎉 处理完成！成功处理 {success}/{total} 张图片")
    else:
        print(f"\n❌ 处理失败！请检查文件夹路径和图片文件")


if __name__ == "__main__":
    main()