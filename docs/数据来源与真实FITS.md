# 数据来源与真实数据口径（REFERENCE）

> 原则：真实数据只用于它真实的环节；没有离焦序列时，不把单帧 FITS 冒充 PD。
> 当前架构：PySide6 桌面 + worker 子进程，真实文件通过原生文件对话框传入，
> 不再有 Web 上传接口（旧 `/api/*` 已随 Web 删除）。

## 1. 真实 FITS（CCD → PSF）

- 桌面入口：① CCD/PSF 页签选择 FITS 文件（worker 任务 `fits_paths`，本地路径直传）；
- 服务入口：`products/ccd/fits_service.py` 的 `run_fits_pipeline`；
- 已测试样例：NASA/STScI HST WFPC2 FITS
  （`https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits`，684 KB、200×200，
  3-D cube 取首层），读到星点、扣背景、输出 64×64 PSF；
- SDSS `frame` FITS 走同一路径。单帧是某波段曝光，**没有离焦序列**，
  只能做 PSF 提取，不能直接做相位差反演。

## 2. 多帧/离焦数据

真实相位差需要同一星、同一光路、不同离焦的多帧；公开 SDSS 不提供这种扫描。

当前方案：

- 合成多离焦（`products.ccd.service.run_ccd_pipeline`）→ PD；
- 真实单帧 FITS → 只做 PSF 提取；
- 真实多帧 FITS 上传/传入时，每帧按自己的星点提 PSF，需保证同星/同视场；
- 没有真实离焦序列时，不声称 PD 精度；
- 采集链路（products/acquisition/preprocess/pipeline）落地后，真实多帧由厂商
  相机接口按高度角扫描采集，当前为标注 mock。

## 3. FEA

没有可直接使用的 2.5 m 望远镜开源 FEA 文件。

- `data/fea_cases/` 的合成 case 由设计资料生成（`synthetic=true`）；
- 真实 FEA 导出 CSV（`x_mm,y_mm,dz_nm`）通过 ③B FEA 页签上传（worker `fea_upload`），
  走 `products/sensitivity/service.py` 的 `custom_surface` 拟合 66 项；
- 真实 FEA 必须来自已收敛的有限元模型，不是用 Zernike 反推；
- 当前真实结果：Abaqus 板级/设计版（FEA-PROVISIONAL），见实测报告。

## 4. Zemax

- 结构：`products/zemax/zmx_parser.py` 解析 ZMX（处方，不是 S）；
- 真实 S：Windows/OpticStudio + ZOS-API（`scripts/zos_sensitivity.py`），
  结果落在 `data/zemax/sensitivity_zemax.json`；
- 没有 license 时只能用已导出的 S CSV/JSON，或明确报“无真实 S”；
- 控制页签 `source="auto"` 会优先真实文件，代理矩阵标注 UNVERIFIED。

## 5. 文档口径

- 合成：`synthetic=true`，结果带 `SYNTHETIC-VALIDATED`；
- 真实 FITS：响应 `real=true`，但仍只是 PSF 提取；
- FEA 上传：`synthetic=false`，标签 FEA-PROVISIONAL（模型限制照写）；
- Zemax 结构解析：`status=parsed`，不是 S；
- 采集 mock：`MOCK-ACQUISITION`/`MOCK-NOT-VENDOR`，不得进结论；
- 任何“精度/灵敏度”结论必须写数据来源和工况。
