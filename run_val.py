import argparse
from val import validation


def main():
    results = validation(
        gt_dir=r"C:\Users\Administrator\Desktop\SSI\label\gt",
        # gt_dir=r"C:\Users\Administrator\Desktop\SSI\label\gt_all",
        pred_dir=r"C:\Users\Administrator\Desktop\SSI\label\pred",
        # pred_dir=r"C:\Users\Administrator\Desktop\SSI\label\pred_all",
        # pred_dir=r"C:\Users\Administrator\Desktop\SSI\label\sahi_pred",
        image_dir=r"C:\Users\Administrator\Desktop\SSI\label\img",
        # image_dir=r"C:\Users\Administrator\Desktop\SSI\label\img_all",
        class_map='seed.json',
        output_file='C:/Users/Administrator/Desktop/SSI/result.json',
        max_detections=120
    )

    # 打印结果 - 使用 results 而不是 metrics

    if results is not None:
        # 打印所有结果
        print("\n验证完成！")
    else:
        print("警告: 未获取到验证结果")

    return results


if __name__ == "__main__":
    main()
