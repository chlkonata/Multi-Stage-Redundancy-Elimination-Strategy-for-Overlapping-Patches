import matplotlib.pyplot as plt

# 1. 准备指定的数据
# 移除了 边缘幼苗 (Class 2)
class_names = ['健康幼苗', '单叶幼苗']
count_values = [10739, 339]

# 2. 设置绘图样式与中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

plt.figure(figsize=(10, 6))

# 3. 绘制柱状图 (保留原有样式：天蓝色、宽度0.5)
bars = plt.bar(class_names, count_values, width=0.5, color='skyblue')

# 4. 完善图表标签
plt.xlabel('Class ID')
plt.ylabel('Count')
plt.title('幼苗类别分布统计')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 5. 在柱子上方自动标注具体的数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12)

# 6. 显示图表
plt.tight_layout()
plt.show()