"""
plot_sensitivity.py - 生成"Zemax 标定 → 灵敏度矩阵"流程图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 中文字体（WSL 里如果没装中文字体，先跑 sudo apt install fonts-wqy-zenhei）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ============================================================
# 标题
# ============================================================
ax.text(8, 8.5, "从 Zemax 标定到灵敏度矩阵",
        fontsize=22, fontweight='bold', ha='center', va='center')

# ============================================================
# 三个主框的坐标
# ============================================================
box_y = 5.0
box_h = 2.8
box_w = 4.2

# 框 1：Zemax 操作
box1_x = 0.5
# 框 2：提取数据
box2_x = 5.9
# 框 3：填充矩阵
box3_x = 11.3

# ============================================================
# 画三个框
# ============================================================
def draw_box(x, y, w, h, title, lines, color):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color,
                          linewidth=1.5)
    ax.add_patch(box)
    # 标题
    ax.text(x + w/2, y + h - 0.35, title,
            fontsize=14, fontweight='bold', ha='center', va='center')
    # 内容
    for i, line in enumerate(lines):
        ax.text(x + 0.2, y + h - 0.9 - i*0.32, line,
                fontsize=11, ha='left', va='center')

# 框 1
draw_box(box1_x, box_y, box_w, box_h,
         "① Zemax 操作",
         ["改第 3 面参数：",
          "  · 厚度 +1.0  → uz",
          "  · 偏心 X +1.0 → ux",
          "  · 偏心 Y +1.0 → uy",
          "  · 倾斜 X +1.0 → uα",
          "  · 倾斜 Y +1.0 → uβ"],
         '#DAE8FC')  # 浅蓝

# 框 2
draw_box(box2_x, box_y, box_w, box_h,
         "② 提取 Zernike 系数",
         ["打开 Zernike Fringe 窗口",
          "",
          "读取 Z4, Z7, Z8 变化量",
          "",
          "Zemax Z7(垂直彗差) → 代码 Z9",
          "Zemax Z8(水平彗差) → 代码 Z8"],
         '#D5E8D4')  # 浅绿

# 框 3
draw_box(box3_x, box_y, box_w, box_h,
         "③ 填充灵敏度矩阵",
         ["每个自由度 → 一列",
          "每个像差 → 一行",
          "",
          "z = IM · u",
          "",
          "单位：λ / 单位调节量"],
         '#FFE6CC')  # 浅橙

# ============================================================
# 三个框之间的箭头
# ============================================================
arrow1 = FancyArrowPatch((box1_x + box_w + 0.05, box_y + box_h/2),
                          (box2_x - 0.05, box_y + box_h/2),
                          arrowstyle='->', mutation_scale=25,
                          linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((box2_x + box_w + 0.05, box_y + box_h/2),
                          (box3_x - 0.05, box_y + box_h/2),
                          arrowstyle='->', mutation_scale=25,
                          linewidth=2, color='black')
ax.add_patch(arrow2)

# ============================================================
# 下方：具体数据示例
# ============================================================
ax.text(8, 4.0, "具体数据示例（uz = 厚度 +1.0 时）",
        fontsize=15, fontweight='bold', ha='center', va='center')

# 三个小框
small_y = 1.8
small_h = 1.7
small_w = 4.2

# 小框 1
draw_box(box1_x, small_y, small_w, small_h,
         "Zemax 读出",
         ["Z4 = -40.398",
          "Z7 =   0.000",
          "Z8 =   0.000"],
         '#F5F5F5')

# 小框 2
draw_box(box2_x, small_y, small_w, small_h,
         "填入矩阵第 1 列",
         ["Z4 行 ← -40.398",
          "Z8 行 ←   0.000",
          "Z9 行 ←   0.000"],
         '#F5F5F5')

# 小框 3
draw_box(box3_x, small_y, small_w, small_h,
         "矩阵片段",
         ["⎡ -40.398  ⋯  ⎤",
          "⎢    0.000  ⋯  ⎥",
          "⎣    0.000  ⋯  ⎦"],
         '#F5F5F5')

# 小框之间的箭头
arrow3 = FancyArrowPatch((box1_x + small_w + 0.05, small_y + small_h/2),
                          (box2_x - 0.05, small_y + small_h/2),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=1.5, color='gray')
ax.add_patch(arrow3)

arrow4 = FancyArrowPatch((box2_x + small_w + 0.05, small_y + small_h/2),
                          (box3_x - 0.05, small_y + small_h/2),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=1.5, color='gray')
ax.add_patch(arrow4)

# ============================================================
# 最下方：最终矩阵
# ============================================================
ax.text(8, 1.0, "最终 3×5 灵敏度矩阵（单位：λ / 0.1 单位调节量）",
        fontsize=14, fontweight='bold', ha='center', va='center')

# 用 text 画矩阵
matrix_text = (
    "          uz       ux       uy       uα       uβ\n"
    "Z4    -4.040   -0.003   -0.003   +0.098   +0.098\n"
    "Z8     0.000   -0.463    0.000    0.000   +6.782\n"
    "Z9     0.000    0.000   -0.463   -6.782    0.000"
)
ax.text(8, 0.3, matrix_text,
        fontsize=11, ha='center', va='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#FFF2CC', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig("sensitivity_matrix_diagram.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("已保存: sensitivity_matrix_diagram.png")