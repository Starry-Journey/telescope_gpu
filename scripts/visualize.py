"""
visualize.py - 可视化 forward_generate 的输出和 inverse_solve 的反演结果
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# ============================================================
# 参数
# ============================================================
THETA = 45
DIR = f"theta_{THETA:03d}deg"
DEFOCUS = [0.0, 1.0, -1.0, 2.0, -2.0]
IMG_SIZE = 256

# ============================================================
# 图 1：5 帧星点图（含噪声）
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))
for i, d in enumerate(DEFOCUS):
    fn = f"{DIR}/star_field_theta{THETA:03d}_defocus_{d:+.1f}waves.fits"
    img = fits.getdata(fn)
    im = axes[i].imshow(img, cmap='inferno', origin='lower',
                        vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.9))
    axes[i].set_title(f"defocus = {d:+.1f}λ", fontsize=13)
    plt.colorbar(im, ax=axes[i], fraction=0.046)
plt.suptitle(f"Star Field Images (θ={THETA}°)", fontsize=15)
plt.tight_layout()
plt.savefig("fig1_star_fields.png", dpi=150, bbox_inches='tight')
plt.close()
print("已保存: fig1_star_fields.png")

# ============================================================
# 图 2：从 FITS 里反推 PSF（星点图归一化后 = PSF 与星点的卷积）
# 简化：选取一个较亮的星点 patch 做展示
# ============================================================
# 找一个较亮星点
img0 = fits.getdata(f"{DIR}/star_field_theta{THETA:03d}_defocus_+0.0waves.fits")
# 扣背景
bg = np.median(img0)
img0_clean = img0 - bg
# 找峰值位置
peak_y, peak_x = np.unravel_index(np.argmax(img0_clean), img0_clean.shape)
# 提取 64x64 patch
half = 32
y1, y2 = max(0, peak_y-half), min(IMG_SIZE, peak_y+half)
x1, x2 = max(0, peak_x-half), min(IMG_SIZE, peak_x+half)

fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))
for i, d in enumerate(DEFOCUS):
    fn = f"{DIR}/star_field_theta{THETA:03d}_defocus_{d:+.1f}waves.fits"
    img = fits.getdata(fn)
    patch = img[y1:y2, x1:x2] - np.median(img[y1:y2, x1:x2])
    im = axes[i].imshow(patch, cmap='inferno', origin='lower')
    axes[i].set_title(f"PSF-like patch (defocus = {d:+.1f}λ)", fontsize=12)
    plt.colorbar(im, ax=axes[i], fraction=0.046)
plt.suptitle(f"Star PSF Patches (θ={THETA}°)", fontsize=15)
plt.tight_layout()
plt.savefig("fig2_psf_patches.png", dpi=150, bbox_inches='tight')
plt.close()
print("已保存: fig2_psf_patches.png")

# ============================================================
# 图 3：Zernike 系数对比（真值 vs 反演）
# ============================================================
if os.path.exists("inverse_result.txt"):
    data = np.loadtxt("inverse_result.txt")
    t4, t8, t9, r4, r8, r9 = data

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Z4\n(Defocus)', 'Z8\n(ComaX)', 'Z9\n(ComaY)']
    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width/2, [t4, t8, t9], width,
                   label='True', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x + width/2, [r4, r8, r9], width,
                   label='Recovered', color='#E63946', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Zernike coefficient (λ)', fontsize=12)
    ax.set_title(f'Zernike Coefficient Comparison (θ={THETA}°)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    err4 = abs(t4 - r4); err8 = abs(t8 - r8); err9 = abs(t9 - r9)
    rms = np.sqrt((err4**2 + err8**2 + err9**2) / 3)
    ax.text(0.98, 0.95, f'RMS = {rms:.4f} λ',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("fig3_zernike_compare.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: fig3_zernike_compare.png")
else:
    print("未找到 inverse_result.txt，跳过 Zernike 对比图")
    print("请先在 inverse_solve.cu 末尾加入保存代码并重新运行")

# ============================================================
# 图 4：残差图 —— 真值 PSF 与反演 PSF 的差异
# 用 Python 端重新生成 PSF 对比（不用 C++，避免依赖）
# ============================================================
def zernike_polys(rho, theta, mask):
    r = rho; t = theta
    Z = {}
    Z[2] = 2 * r * np.cos(t)
    Z[3] = 2 * r * np.sin(t)
    Z[4] = np.sqrt(3) * (2*r**2 - 1)
    Z[5] = np.sqrt(6) * r**2 * np.sin(2*t)
    Z[6] = np.sqrt(6) * r**2 * np.cos(2*t)
    Z[7] = np.sqrt(8) * r**3 * np.sin(3*t)
    Z[8] = np.sqrt(8) * (3*r**3 - 2*r) * np.cos(t)
    Z[9] = np.sqrt(8) * (3*r**3 - 2*r) * np.sin(t)
    for k in Z:
        Z[k] = Z[k] * mask
    return Z

def compute_psf(coeffs, defocus, pupil_radius=0.45, fft_size=512, img_size=256, oversample=2):
    """与 C++ 版完全一致的 PSF 计算"""
    y, x = np.mgrid[-1:1:fft_size*1j, -1:1:fft_size*1j]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    mask = (rho <= pupil_radius).astype(float)
    r = np.clip(rho / pupil_radius, 0, 1)
    t = theta
    Z = zernike_polys(r, t, mask)

    # 波前
    wf = np.zeros_like(rho)
    for i, k in enumerate([2,3,4,5,6,7,8,9]):
        c = coeffs[i]
        if k == 4: c += defocus
        wf += c * Z[k]

    pupil = mask.astype(np.complex128) * np.exp(1j * 2 * np.pi * wf)
    psf_fft = np.fft.fftshift(np.fft.fft2(pupil))
    psf = np.abs(psf_fft)**2
    psf /= psf.sum()

    # 降采样
    psf_down = psf.reshape(img_size, oversample, img_size, oversample).sum(axis=(1,3))
    psf_down /= psf_down.sum()
    return psf_down

if os.path.exists("inverse_result.txt"):
    data = np.loadtxt("inverse_result.txt")
    t4, t8, t9, r4, r8, r9 = data

    defocus_list = [0.0, 1.0, -1.0, 2.0, -2.0]
    fig, axes = plt.subplots(3, 5, figsize=(25, 14))

    for j, d in enumerate(defocus_list):
        # 真值 PSF
        true_coeffs = np.array([0, 0, t4, 0, 0, 0, t8, t9])
        psf_true = compute_psf(true_coeffs, d)

        # 反演 PSF
        rec_coeffs = np.array([0, 0, r4, 0, 0, 0, r8, r9])
        psf_rec = compute_psf(rec_coeffs, d)

        # 残差
        diff = psf_true - psf_rec

        # 第 1 行：真值
        im1 = axes[0, j].imshow(np.log10(psf_true + 1e-10), cmap='inferno', origin='lower')
        axes[0, j].set_title(f"True (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im1, ax=axes[0, j], fraction=0.046)

        # 第 2 行：反演
        im2 = axes[1, j].imshow(np.log10(psf_rec + 1e-10), cmap='inferno', origin='lower')
        axes[1, j].set_title(f"Recovered (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im2, ax=axes[1, j], fraction=0.046)

        # 第 3 行：残差
        vmax = np.max(np.abs(diff))
        im3 = axes[2, j].imshow(diff, cmap='seismic', origin='lower',
                                 vmin=-vmax, vmax=vmax)
        axes[2, j].set_title(f"Residual (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im3, ax=axes[2, j], fraction=0.046)

    axes[0, 0].set_ylabel('True', fontsize=13)
    axes[1, 0].set_ylabel('Recovered', fontsize=13)
    axes[2, 0].set_ylabel('Residual', fontsize=13)

    plt.suptitle(f'PSF Comparison: True vs Recovered vs Residual (θ={THETA}°)',
                 fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig("fig4_psf_residual.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: fig4_psf_residual.png")

print("\n所有可视化完成！")