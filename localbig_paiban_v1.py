import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# ==========================================
# 👇 用户配置区域 (请在这里修改字号)
# ==========================================
# 1. 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

# 2. 设置底部标签的字号 (数字越大，字越大)
LABEL_FONT_SIZE = 18  # 👈 您可以在这里改为 10, 12, 16, 18 等
LABEL_WEIGHT = 'bold'  # 👈 如果不想加粗，请改为 'normal'


# ==========================================
# 1. 带调试功能的图片读取函数
# ==========================================
def create_composite_image(image_path, center_red, center_green, crop_size=32):
    full_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        print(f"❌ [文件缺失] 找不到: {image_path}")
        # 生成黑色占位图
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(img, "MISSING", (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    zoom_size = w // 2
    main_img = img.copy()

    def get_top_left(center, size, limit_w, limit_h):
        x = max(0, min(center[0] - size // 2, limit_w - size))
        y = max(0, min(center[1] - size // 2, limit_h - size))
        return x, y

    rx, ry = get_top_left(center_red, crop_size, w, h)
    gx, gy = get_top_left(center_green, crop_size, w, h)

    cv2.rectangle(main_img, (rx, ry), (rx + crop_size, ry + crop_size), (255, 0, 0), 2)
    cv2.rectangle(main_img, (gx, gy), (gx + crop_size, gy + crop_size), (0, 255, 0), 2)

    crop_red = img[ry:ry + crop_size, rx:rx + crop_size]
    crop_green = img[gy:gy + crop_size, gx:gx + crop_size]

    zoom_red = cv2.resize(crop_red, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
    zoom_green = cv2.resize(crop_green, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)

    border_w = 4
    cv2.rectangle(zoom_red, (0, 0), (zoom_size - 1, zoom_size - 1), (255, 0, 0), border_w)
    cv2.rectangle(zoom_green, (0, 0), (zoom_size - 1, zoom_size - 1), (0, 255, 0), border_w)

    footer = np.hstack((zoom_red, zoom_green))
    composite = np.vstack((main_img, footer))

    return composite


# ==========================================
# 2. 主程序
# ==========================================
def plot_full_grid_custom_font():
    row_keys = ["T1", "T2", "Gad"]
    col_map = [
        # ("MRI", "(a) MRI"), ("PET", "(b) PET"),
        ("MRI", "(a) MRI"), ("SPECT", "(b) SPECT"),
        ("EMMA", "(c) EMMA"), ("MIF_O", "(d) MIF_O"),
        ("MM_Net", "(e) MM_Net"), ("GeSeNet", "(f) GeSeNet"),
        ("Swin_F", "(g) Swin_F"), ("MMAE", "(h) MMAE"),
        ("LFDT", "(i) LFDT"), ("GIFNet", "(j) GIFNet"),
        ("M4FNet", "(k) M4FNet"), ("Ours", "(l) Ours")
    ]

    # ⚠️ 请确保这里是您真实的 ROI 坐标
    # MRI-PET
    # roi_configs = [
    #     {"red": (80, 80), "green": (190, 130)},
    #     {"red": (80, 80), "green": (160, 130)},
    #     {"red": (128, 128), "green": (90, 205)}
    # ]

    # MRI-SPECT
    roi_configs = [
        {"red": (70, 170), "green": (160, 175)},
        {"red": (80, 175), "green": (160, 130)},
        {"red": (135, 128), "green": (90, 195)}
    ]
    rows = len(row_keys)
    cols = len(col_map)

    # 布局参数
    margin_left = 0.01
    margin_right = 0.99
    margin_top = 0.98
    margin_bottom = 0.06  # 稍微增加底部留白，防止大字体被切掉
    w_space = 0.0
    h_space = 0.02

    # 计算完美尺寸
    single_img_w = 256
    single_img_h = 384
    img_aspect = single_img_w / single_img_h
    fig_width = 24
    effective_w_frac = margin_right - margin_left
    effective_h_frac = margin_top - margin_bottom
    fig_height = (fig_width * effective_w_frac * rows) / (cols * effective_h_frac * img_aspect)

    print(f"画布尺寸: {fig_width} x {fig_height:.2f}")

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=w_space, hspace=h_space,
                        left=margin_left, right=margin_right,
                        top=margin_top, bottom=margin_bottom)

    print("--- 开始生成 ---")

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            filename = f"imgs/{row_keys[r]}_{col_map[c][0]}.png"
            composite = create_composite_image(filename, roi_configs[r]["red"], roi_configs[r]["green"])

            ax.imshow(composite, aspect='equal')
            ax.axis('off')

            if r == rows - 1:
                # 使用您在顶部定义的字号
                ax.text(0.5, -0.05, col_map[c][1], transform=ax.transAxes,
                        va='top', ha='center',
                        fontsize=LABEL_FONT_SIZE,  # 👈 应用自定义字号
                        fontweight=LABEL_WEIGHT,  # 👈 应用自定义粗细
                        fontname='Times New Roman')

    output_file = "Fig11_Custom_Font_SPECT.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 完成！图片保存为: {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_full_grid_custom_font()