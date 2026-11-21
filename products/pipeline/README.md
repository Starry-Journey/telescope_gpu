# products/pipeline —— 端到端数据管线编排

> 把“俯仰角扫描 → 采集 → 预处理 → PD 反演 → 实测 S → 补偿表 → FEA/装调请求”
> 串成一次可复现运行；硬件只通过 acquisition 的抽象接口访问。

## 职责

- `PipelineConfig` 固定一次观测的参数（俯仰角列表、离焦栈、曝光、PD 项、
  激励幅值/重复次数、视场/像元比例、seed、定标目录、`stop_on_error`、`keep_frames`）；
- `WavefrontPipeline.run()` 逐俯仰角：指向 → 机构回零 → 逐 DOF 括阻激励 →
  每位姿 capture → preprocess → PD → `measure_sensitivity` → 有界最小二乘补偿；
- 用 `analyze_control_space` 找不可补偿项，生成 `fea_request`；记录 stages 耗时、
  warnings 与 manifest（schema v1）。

## 输入 / 输出

- 输入：`TelescopeTerminalApi` / `CameraApi` / `MechanismApi` 的实现（mock 或真实 adapter）
  + `PipelineConfig`；
- 输出（JSON 安全 dict）：`label`/`mode`/`stages`/`per_elevation`/
  `compensation_table`/`fea_request`/`warnings`/`manifest`；
- 结果不保留原始帧（`keep_frames=False` 默认），内存有界。

## 关键文件与函数

| 文件 | 关键接口 |
|---|---|
| `products/pipeline/orchestrator.py` | `PipelineConfig`、`WavefrontPipeline`（`__init__`/`run`）、`run_pipeline`、`MANIFEST_SCHEMA_VERSION` |
| `products/pipeline/__init__.py` | 导出上述公共 API |

## 依赖

- numpy；复用 `products.acquisition`、`products.preprocess`、
  `products.pd.service`、`products.sensitivity.measured`、
  `products.sensitivity.observability`；
- 不依赖 GUI/Qt/网络；不 import `products.control` 的 GUI 服务（补偿用
  `products.lsq.bounded_least_squares` / scipy `lsq_linear` 解有界最小二乘）。

## 标签与验证状态

- 默认 mock 全链：SYNTHETIC-VALIDATED（`mode=mock`），只证明链路可跑，不是实测；
- `fea_request` 是**前处理请求**，不是 FEA 结果；真实 FEA/装调需另跑；
- 真实台架：UNVERIFIED（无硬件 API/工况数据）。

## 常见坑

1. 一次只处理一个俯仰角序列；`Stages`/`per_elevation` 是诊断用，别当科学结论；
2. `keep_frames=True` 会保留每帧，长扫描内存暴涨；只在调试时开；
3. 没有真实定标目录时 `calibration="synthetic"`（用 mock 相机合成 master），必须如实上报；
4. 补偿量是“最小范数 + 边界投影”的解，零空间方向不可观测，不能当物理真值；
5. `stop_on_error=True` 时任一帧/位姿失败会中止该次运行；批处理要显式设 False 并看 warnings。
