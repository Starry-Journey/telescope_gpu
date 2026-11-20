"""端到端数据管线（pipeline）：接口驱动的采集 -> 反演 -> 补偿 -> FEA 请求。

输入:
    - :class:`products.acquisition.interfaces` 的三个实现（mock 或真实
      adapter）+ :class:`PipelineConfig`。
输出:
    - :func:`WavefrontPipeline.run` 返回 JSON 安全 dict：每个俯仰角的补偿量、
      补偿表、不可补偿项（FEA/装调请求）、阶段耗时、manifest（schema v1）。
边界:
    - 不依赖 GUI/Qt/网络；PD 反演复用 products.pd.service，灵敏度复用
      products.sensitivity.measured，可观测性复用
      products.sensitivity.observability；
    - 没有真实工况数据，也不运行 FEA——``fea_request`` 是请求而非结果。
"""
from __future__ import annotations

from .orchestrator import (MANIFEST_SCHEMA_VERSION, PipelineConfig,
                           WavefrontPipeline, run_pipeline)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PipelineConfig",
    "WavefrontPipeline",
    "run_pipeline",
]
