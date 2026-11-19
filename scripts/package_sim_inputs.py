r"""Package the simulation inputs (FEA CSVs + ZMX + Zemax S) into one folder.

The folder is a self-contained hand-off for engineers / the desktop app:

  model/2.5m主系统.zmx            real optical prescription (ZOS-API source)
  fea/*.csv                       Abaqus top-surface exports (base + k-sweep)
  fea/manifest.json               case registry (labels, RMS, Zernike)
  fea/fit_summary.json            Zernike fits of the base cases
  fea/sensitivity_k_sweep.json    S interval over the support-stiffness sweep
  zemax/sensitivity_zemax.*       real 3x5 M2 DOF matrix
  zemax/sensitivity_zemax_scope.json  step/field/wavelength applicability study
  README.md                       what each file is, and how to load it

Run:
  py -3.13 scripts\package_sim_inputs.py
"""
from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZMX_CANDIDATES = [
    Path(r"C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统.zmx"),
    Path(r"C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统_1.ZMX"),
]


def copy_tree(src: Path, dst: Path, patterns=("*",)):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in patterns:
        for f in sorted(src.glob(pat)):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                n += 1
    return n


def main() -> int:
    stamp = date.today().isoformat()
    out = ROOT / "release" / ("telescope_sim_inputs_" + stamp)
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "fea").mkdir(parents=True)
    (out / "zemax").mkdir(parents=True)

    zmx = next((p for p in ZMX_CANDIDATES if p.exists()), None)
    if zmx is None:
        print("WARNING: no ZMX found; package will miss the optical model")
    else:
        shutil.copy2(zmx, out / "model" / "2.5m主系统.zmx")
        print("model:", zmx.name, zmx.stat().st_size, "bytes")

    fea_src = ROOT / "data" / "fea_cases"
    n_csv = copy_tree(fea_src / "abaqus_m1", out / "fea", ("*_top.csv",))
    for name in ("manifest.json", "fit_summary.json", "sensitivity_k_sweep.json",
                 "engineering_terms_sensitivity.json"):
        src = (fea_src / "abaqus_m1" / name) if name != "manifest.json" else (fea_src / name)
        if src.exists():
            shutil.copy2(src, out / "fea" / name)
    print("fea csv:", n_csv)

    zemax_src = ROOT / "data" / "zemax"
    n_z = copy_tree(zemax_src, out / "zemax",
                    ("sensitivity_zemax.json", "sensitivity_zemax.csv",
                     "sensitivity_zemax_scope.json"))
    print("zemax files:", n_z)

    readme = """# TelescopeEngineering 仿真前置数据包

生成日期: {stamp}
来源: 许可版 Ansys Zemax OpticStudio 2023 R1 (ZOS-API) + Abaqus 2025

## 目录

```
model/2.5m主系统.zmx              真实光学处方（ZOS-API 求解输入）
fea/*_top.csv                     Abaqus 顶面位移导出 (x_mm,y_mm,z_mm,u1_mm,u2_mm,dz_nm)
fea/manifest.json                 工况注册表: id/载荷/口径/RMS/Zernike/label
fea/fit_summary.json              基础工况的 66 项 Zernike 拟合摘要
fea/sensitivity_k_sweep.json      支撑刚度扫描 (3e5/1e6/3e6 N/mm) 的 S 区间
zemax/sensitivity_zemax.json      真实 3x5 M2 5-DOF 灵敏度矩阵
zemax/sensitivity_zemax.csv       同上，CSV 形式
zemax/sensitivity_zemax_scope.json 步长/视场/波长适用范围研究
```

## 怎么用

1. 桌面软件读取的默认路径是仓库的 `data/fea_cases/` 与 `data/zemax/`。
   若要复现本包，把 `fea/` 下的 CSV 和 `manifest.json` 放回
   `data/fea_cases/abaqus_m1/` 与 `data/fea_cases/`，把 `zemax/` 放回
   `data/zemax/`。
2. 只做面形/灵敏度复算：
   `python scripts/abaqus_m1_fit.py`（拟合 + 环境 S）
   `python scripts/abaqus_k_sweep_summary.py`（刚度区间）
3. 重新生成 Zemax S（需要许可与 ZMX）：
   `py -3.13 scripts/zos_sensitivity.py --zmx model/2.5m主系统.zmx --out data/zemax`

## 标签与限定

- Zemax 矩阵: EXTERNAL-CHECK，波长 0.7 µm、单视场、Z4/Z8/Z9 三项、线性小量。
  步长 0.005–0.02 内相对变化 6e-7（严格线性）；波长换到 0.4 µm 时 S 按 1/λ
  缩放（以波为单位），改用 nm 表示则与波长无关。
- Abaqus 结果: FEA-PROVISIONAL。板级模型、弹簧垫等效支撑，绝对量级依赖
  支撑刚度假设，请引用 `sensitivity_k_sweep.json` 的区间而非单值。
- 本包不含实测数据；实测 S 由 `products/sensitivity/measured.py` 的
  CCD+PD 流程产生，Zemax/FEA 只用于实验设计与交叉验证。
""".format(stamp=stamp)
    (out / "README.md").write_text(readme, encoding="utf-8")

    zpath = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out.parent))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print("\nPACKAGE:", out)
    print("ZIP:    ", zpath, "%.1f MB" % (zpath.stat().st_size / 1e6))
    print("files:  ", len([f for f in out.rglob('*') if f.is_file()]),
          "%.2f MB" % (total / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
