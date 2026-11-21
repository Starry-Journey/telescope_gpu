"""
visualize.py - 可视化 forward_generate 的输出和 inverse_solve 的反演结果

用法（在 build/ 下运行）：
    python3 ../scripts/visualize.py --theta 45
    python3 ../scripts/visualize.py --theta 45 --data-dir theta_045deg \
        --result inverse_result.txt --out-dir figures

依赖：numpy、matplotlib、astropy。图 1/2/4 只依赖 FITS 文件，图 3 需要
inverse_result.txt（6 个数：真值 Z4/Z8/Z9 + 反演 Z4/Z8/Z9）。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

# 仓库根锚定，保证从任意工作目录运行时都能 import products.*
# （与 scripts/benchmark.py 的处理一致）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from products.optics.forward import forward_psf_numpy  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=45.0,
                        help="pitch angle in degrees (default: 45)")
    parser.add_argument("--data-dir", default=None,
                        help="FITS directory (default: theta_<theta>deg)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="detector size in pixels (default: 256)")
    parser.add_argument("--defocus", default="0,1,-1,2,-2",
                        help="comma separated defocus values in waves")
    parser.add_argument("--result", default="inverse_result.txt",
                        help="6-number result file written by inverse_solve")
    parser.add_argument("--out-dir", default=None,
                        help="figure output directory (default: current dir)")
    return parser.parse_args()


def frame_path(data_dir, theta, defocus):
    return data_dir / f"star_field_theta{int(round(theta)):03d}_defocus_{defocus:+.1f}waves.fits"


def load_frame(data_dir, theta, defocus, img_size):
    path = frame_path(data_dir, theta, defocus)
    if not path.exists():
        raise FileNotFoundError(
            f"missing FITS: {path}\n"
            f"run: ./forward_generate --theta {theta:g} --img-size {img_size}")
    image = np.asarray(fits.getdata(str(path)), dtype=np.float64)
    if image.shape != (img_size, img_size):
        print(f"warning: {path.name} shape {image.shape} != ({img_size}, {img_size})")
    return image


def robust_limits(image):
    lo = np.percentile(image, 1.0)
    hi = np.percentile(image, 99.9)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    return lo, hi


def plot_star_fields(args, defocus, data_dir, out_dir):
    fig, axes = plt.subplots(1, len(defocus), figsize=(5 * len(defocus), 5.5))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, defocus):
        img = load_frame(data_dir, args.theta, d, args.img_size)
        vmin, vmax = robust_limits(img)
        im = ax.imshow(img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"defocus = {d:+.1f}λ", fontsize=13)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Star Field Images (θ={args.theta:g}°)", fontsize=15)
    fig.tight_layout()
    path = out_dir / "fig1_star_fields.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_psf_patches(args, defocus, data_dir, out_dir):
    # 以焦面帧中最亮的星为中心，截取所有离焦帧的同一块 patch。
    img0 = load_frame(data_dir, args.theta, defocus[0], args.img_size)
    bg = np.median(img0)
    peak_y, peak_x = np.unravel_index(np.argmax(img0 - bg), img0.shape)
    half = max(8, min(32, args.img_size // 4))
    y1, y2 = max(0, peak_y - half), min(args.img_size, peak_y + half)
    x1, x2 = max(0, peak_x - half), min(args.img_size, peak_x + half)

    fig, axes = plt.subplots(1, len(defocus), figsize=(5 * len(defocus), 5.5))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, defocus):
        img = load_frame(data_dir, args.theta, d, args.img_size)
        patch = img[y1:y2, x1:x2] - np.median(img[y1:y2, x1:x2])
        im = ax.imshow(patch, cmap="inferno", origin="lower")
        ax.set_title(f"defocus = {d:+.1f}λ", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Star PSF Patches (θ={args.theta:g}°)", fontsize=15)
    fig.tight_layout()
    path = out_dir / "fig2_psf_patches.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_zernike_compare(args, out_dir):
    path = Path(args.result)
    if not path.exists():
        print(f"skip fig3: {path} not found (run inverse_solve first)")
        return False
    values = np.atleast_1d(np.loadtxt(path))
    if values.size != 6:
        print(f"skip fig3: expected 6 numbers in {path}, got {values.size}")
        return False
    t4, t8, t9, r4, r8, r9 = values

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Z4\n(Defocus)", "Z8\n(Coma X)", "Z9\n(Coma Y)"]
    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width / 2, [t4, t8, t9], width,
                   label="True", color="#2E86AB", edgecolor="black")
    bars2 = ax.bar(x + width / 2, [r4, r8, r9], width,
                   label="Recovered", color="#E63946", edgecolor="black")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Zernike coefficient (λ)", fontsize=12)
    ax.set_title(f"Zernike Coefficient Comparison (θ={args.theta:g}°)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    err = np.array([t4 - r4, t8 - r8, t9 - r9])
    coeff_rms = float(np.sqrt(np.mean(err ** 2)))
    ax.text(0.98, 0.95, f"coeff RMS = {coeff_rms:.4f} λ",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    out = out_dir / "fig3_zernike_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    return True


# 本脚本的槽位布局与 C++ src/zernike.cpp 一致：coeffs[slot-2]，slots 2..9。
# PSF 算法核心不再在此复制第三份，统一走 products.optics.forward（单一来源）。
_SLOT_TO_LABEL = {4: "Z4", 5: "Z5", 6: "Z6", 8: "Z8", 9: "Z9"}


def compute_psf(coeffs, defocus, pupil_radius=0.45, img_size=256, oversample=2):
    """残差图用 PSF：薄适配到 products.optics.forward.forward_psf_numpy。

    与旧的本文件第三份实现相比有三处口径差异，处理如下：
      * 槽位布局：共享模型只实现 Z4/Z5/Z6/Z8/Z9（无 tip/tilt/trefoil 槽位
        2/3/7）；非零槽位落在可映射集合内才可计算，否则显式报错而非静默丢弃。
      * sum-binning：旧实现 reshape 后 .sum() 分箱，共享实现 .mean() 分箱，
        但两者随后都做单位和归一化，数学上等价（sum = 4*mean，归一化后相同，
        已实测一致到 ~1e-18）。
      * FFT 归一化：旧实现未用 norm="ortho"，共享实现用了——常数因子同样被
        单位和归一化抵消。mgrid[-1:1:Nj] 与 linspace(-1,1,N) 网格逐点相同。
    """
    active = {slot: float(coeffs[slot - 2]) for slot in _SLOT_TO_LABEL
              if float(coeffs[slot - 2]) != 0.0}
    unsupported = [s for s in range(2, 10)
                   if s not in _SLOT_TO_LABEL and float(coeffs[s - 2]) != 0.0]
    if unsupported:
        raise ValueError(
            f"unsupported nonzero Zernike slot(s) {unsupported}: the shared "
            f"forward model (products.optics) implements Z4/Z5/Z6/Z8/Z9 only")
    if not active:
        # 全零波前：仍经共享实现，使 defocus 继续作用于 Z4（与旧行为一致）
        active = {4: 0.0}
    slots = sorted(active)
    return forward_psf_numpy([active[s] for s in slots], float(defocus),
                             n=img_size, oversample=oversample,
                             pupil_radius=pupil_radius,
                             terms=tuple(_SLOT_TO_LABEL[s] for s in slots))


def plot_psf_residual(args, defocus, out_dir):
    path = Path(args.result)
    if not path.exists():
        print(f"skip fig4: {path} not found")
        return
    values = np.atleast_1d(np.loadtxt(path))
    if values.size != 6:
        print(f"skip fig4: expected 6 numbers in {path}, got {values.size}")
        return
    t4, t8, t9, r4, r8, r9 = values

    fig, axes = plt.subplots(3, len(defocus), figsize=(5 * len(defocus), 14))
    axes = np.atleast_2d(axes)
    for j, d in enumerate(defocus):
        true_coeffs = np.array([0, 0, t4, 0, 0, 0, t8, t9])
        rec_coeffs = np.array([0, 0, r4, 0, 0, 0, r8, r9])
        psf_true = compute_psf(true_coeffs, d, img_size=args.img_size)
        psf_rec = compute_psf(rec_coeffs, d, img_size=args.img_size)
        diff = psf_true - psf_rec

        im1 = axes[0, j].imshow(np.log10(psf_true + 1e-10), cmap="inferno", origin="lower")
        axes[0, j].set_title(f"True (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im1, ax=axes[0, j], fraction=0.046)

        im2 = axes[1, j].imshow(np.log10(psf_rec + 1e-10), cmap="inferno", origin="lower")
        axes[1, j].set_title(f"Recovered (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im2, ax=axes[1, j], fraction=0.046)

        vmax = np.max(np.abs(diff))
        im3 = axes[2, j].imshow(diff, cmap="seismic", origin="lower", vmin=-vmax, vmax=vmax)
        axes[2, j].set_title(f"Residual (defocus={d:+.1f}λ)", fontsize=11)
        plt.colorbar(im3, ax=axes[2, j], fraction=0.046)

    axes[0, 0].set_ylabel("True", fontsize=13)
    axes[1, 0].set_ylabel("Recovered", fontsize=13)
    axes[2, 0].set_ylabel("Residual", fontsize=13)
    fig.suptitle(f"PSF Comparison (θ={args.theta:g}°)", fontsize=15, y=0.995)
    fig.tight_layout()
    out = out_dir / "fig4_psf_residual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir or f"theta_{int(round(args.theta)):03d}deg")
    defocus = [float(v) for v in args.defocus.split(",") if v.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        sys.exit(f"data dir not found: {data_dir}")

    plot_star_fields(args, defocus, data_dir, out_dir)
    plot_psf_patches(args, defocus, data_dir, out_dir)
    plot_zernike_compare(args, out_dir)
    plot_psf_residual(args, defocus, out_dir)
    print("done.")


if __name__ == "__main__":
    main()
