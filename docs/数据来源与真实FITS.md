# 开源/真实数据与模拟数据

> 原则：真实数据只用于它真实的环节；没有离焦序列时，不把单帧 FITS 冒充 PD。

## 1. 真实 FITS（模块 1）

支持：

```text
POST /api/ccd/upload
{"files":[{"name":"WFPC2.fits","data_b64":"..."}],"defocus":[0]}
```

已测试：

- NASA/STScI HST WFPC2 FITS 样例：
  `https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits`
- 684 KB，200×200，3-D cube 取首层；
- 读到星点、扣背景、输出 64×64 PSF。

SDSS `frame` FITS 走同一接口。SDSS 单帧是某一波段曝光，**没有离焦序列**，
因此只能做 PSF 提取，不能直接做相位差反演。

## 2. 多帧/离焦数据

真实相位差需要同一星、同一光路、不同离焦的多帧。公开 SDSS 不提供这种扫描。

当前方案：

- 模块 1 合成多离焦（`psf_model="phase"`）→ 模块 2 PD；
- 真实单帧 FITS → 只做 PSF 提取；
- 真实多帧 FITS 上传时，每帧按自己的星点提取 PSF，需用户保证同星/同视场；
- 没有真实离焦序列时，不声称 PD 精度。

## 3. FEA

没有找到可直接使用的 2.5 m 望远镜开源 FEA 文件。

当前：

- `data/fea_cases/` 是设计资料生成的合成参考工况；
- `POST /api/fea/upload` 支持真实 FEA 导出 `x_mm,y_mm,dz_nm` CSV，拟合 66 项；
- 替换 `data/fea_cases/*.csv` 或调用 upload 即可；
- 真实 FEA 必须来自已收敛的有限元模型，不是用 Zernike 反推。

## 4. Zemax

- `products/zemax/zmx_parser.py` 解析结构；
- 真实 S 需要 ZOS-API（Windows/OpticStudio）；
- 没有 license 时，用 Zemax 导出的 S CSV 走 ⑦；
- 详见 `docs/Zemax3A.md`。

## 5. 文档口径

- 合成：`synthetic=true`，页面黄条；
- 真实：FITS 响应 `real=true`；
- FEA 上传：`synthetic=false`；
- Zemax 结构解析：`status=parsed`，不是 `S`；
- 任何“精度/灵敏度”结论必须写数据来源和工况。
