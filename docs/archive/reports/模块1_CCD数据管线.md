# 模块1：CCD 数据清洗与真实感仿真

> 相位/噪声 → CCD 星场 → 定标 → PSF cutout


## 01 数据链

> 从相位参数到 PSF。

*代码 1-1  数据链*

```cpp
Z4/Z8/Z9 + defocus
  -> Fourier-optics PSF kernel
  -> star field scene
  -> Poisson + dark + flat + bias + read noise
  -> saturation/blooming + cosmic rays
  -> raw FITS
  -> (raw-bias-dark*t)/flat
  -> isolated star cutout 64x64
  -> PSF stack [5,64,64]
```

*表 1-1  实现映射*

| 阶段 | 实现 | 文件 |
|---|---|---|
| 相位 PSF | ml.traditional.forward_psf_numpy | pipeline/realistic.py |
| 星场/噪声 | Moffat/相位 PSF + CCD 效应 | pipeline/realistic.py |
| 定标 | bias/dark/flat | products/ccd/service.py |
| PSF 提取 | 孤立定标星 + 平面背景 + 归一化 | products/ccd/service.py |


## 02 物理与真实感

> 哪些效应被模拟，哪些没有。

*表 2-1  效应清单*

| 效应 | 实现 | 说明 |
|---|---|---|
| Zernike 相位 PSF | Z4/Z8/Z9 + defocus | 与 PD 反演共用同一前向 |
| 大气/衍射 | 可切换 Moffat/phase | phase 模式用于模块2 |
| 噪声 | Poisson + 暗电流 + 读出 | read_noise 可调 |
| 平场/渐晕/灰尘 | flat + vignette + dust | 定标可恢复 |
| 饱和/blooming | full well 溢出 | 亮星保护 |
| 宇宙线 | 轨迹 | 不是单点 |
| 热/坏像素 | hot/dead pixels | 定标/掩膜 |
| CTE/电荷扩散 | 未实现 | 真实感边界 |

PSF 来自孤立定标星，避免饱和/邻居污染；这是“数据管线”演示，不是完整 crowded-field 测光。


## 03 接口与参数

> 一次请求返回 5 帧 + PSF。

*代码 3-1  接口*

```cpp
POST /api/ccd/pipeline
{"c4":0.3,"c8":0.1,"c9":-0.05,"noise":2,"stars":20,
 "defocus":[-2,-1,0,1,2],"seed":3}
-> {"frames":[{raw,cal}], "psf":[[64x64]x5],
    "psf_png":[...], "truth":[c4,c8,c9],
    "star":{x,y,flux}, "metrics":{...}}
```

*表 3-1  参数*

| 参数 | 含义 | 单位 |
|---|---|---|
| c4/c8/c9 | 输入 Zernike | waves |
| noise | 读出噪声 | e- |
| stars | 星数 | — |
| defocus | 离焦序列 | waves |
| seed | 随机种子 | — |


## 04 实测效果

> 合成 CCD + 5 个 PSF。

![图 4-1  定标后的 CCD 星场与 5 个离焦 PSF cutout。](figs/fig_module1_ccd_psf.png)

*图 4-1  定标后的 CCD 星场与 5 个离焦 PSF cutout。*

当前示例：c4=0.3、c8=0.1、c9=-0.05、read noise=2 e-、192²、5 帧。PSF cutout 为 64×64，直接送模块 2。


### 4.1 真实 FITS 测试

新增 POST /api/ccd/upload，可直接读取已定标 FITS（SDSS/HST/公开归档）。前端文件选择后转 base64；后端用 astropy 读取 2-D/3-D 图像、找孤立星、扣背景、提取 64×64 PSF。

*代码 4-1  真实 FITS 接口*

```cpp
POST /api/ccd/upload
{"files":[{"name":"WFPC2.fits","data_b64":"..."}],
 "defocus":[0]}
-> {"psf":[[64x64]], "star":{x,y,flux},
    "metrics":{real:true, background, noise, shape}}

```

用 NASA FITS 样例 WFPC2u5780205r_c0fx.fits（684 KB，200×200）实测：读取成功，检测到星点，输出 1 个 PSF cutout。单帧只能做 PSF 提取，不能做 PD；PD 需要同一星的多帧不同离焦。


## 05 边界

> 不要把合成数据写成实测。

- 当前不是实测 CCD，也没有真实观测的 WCS/导星；
- PSF 来自孤立定标星，crowded field 需要额外去混叠；
- 没有 CTE/电荷扩散/非线性/持久性；
- 真实数据要接 FITS 上传、bias/dark/flat 产品、坏像素掩膜；
- UI 的“真实感”是视觉/噪声层面，不是 phosim 完整光路。
