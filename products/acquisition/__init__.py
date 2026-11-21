"""数据采集层（acquisition）：真实硬件边界 + 显式 mock。

本包把新一代端到端数据管线的三个外部边界固定下来：

* :mod:`products.acquisition.interfaces` —— 望远镜终端 / 天文相机 /
  补偿机构的抽象接口、单位约定、时序与错误码要求；
* :mod:`products.acquisition.mock` —— 今天唯一可运行的实现，所有对象都
  带 ``mode="mock"``、``label="SYNTHETIC-VALIDATED"``，用于端到端联调。

输入:
    - 无（接口层）；mock 由构造参数（seed / 真值矩阵 / 运动学参数）驱动。
输出:
    - 抽象方法契约；mock 返回 raw ADU 帧、指向/工况遥测、机构位置。
边界:
    - **没有望远镜控制 API，也没有工况（operating condition）数据**；
      真实 SDK 到位后，在独立的 adapter 模块里实现这些接口，不要改本包
      的接口语义。mock 里出现的一切物理量都是合成占位，不能当作实测证据。

``products.acquisition.mock`` 故意不在包的 ``__init__`` 里导入：接口模块
本身应当保持轻量（不依赖 products.ccd / matplotlib），需要 mock 的
调用方显式 ``from products.acquisition.mock import build_mock_rig``。
"""
from __future__ import annotations

from .interfaces import (AcquisitionError, ApiTimeoutError, CameraApi,
                         HardwareError, MechanismApi, TelescopeTerminalApi,
                         UNIT_SYSTEM, WAVELENGTH_NM)

__all__ = [
    "AcquisitionError",
    "ApiTimeoutError",
    "CameraApi",
    "HardwareError",
    "MechanismApi",
    "TelescopeTerminalApi",
    "UNIT_SYSTEM",
    "WAVELENGTH_NM",
]
