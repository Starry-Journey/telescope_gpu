"""数据采集层抽象接口（真实硬件边界的唯一契约）。

本模块定义新一代端到端数据管线的三个外部边界：望远镜控制终端、天文相机
终端、补偿机构控制系统。**今天没有可用的供应商 API，也没有工况
（operating condition）数据**，所以这里只放抽象接口；真实实现必须在
独立的 adapter 模块里完成，并遵守下面每个类/方法的单位、时序和错误码
约定。

输入（抽象基类）:
    - 无。子类由 adapter 模块（真实 SDK）或 :mod:`products.acquisition.mock`
      （今天的显式 mock，label="SYNTHETIC-VALIDATED"）提供。
输出:
    - 方法签名、单位约定、遥测字段要求、异常/错误码约定。
边界（boundaries）:
    - 本模块不 import 任何供应商 SDK、不发起网络/串口/PLC 调用；
    - 真实 adapter 必须保证：
        1. 单位：对外只使用本模块文档中的 units-of-record（内部换算在
           adapter 内完成，绝不把供应商原始计数泄漏给管线）；
        2. 时序：``move``/``goto_*`` 阻塞到"已到位且稳定"才返回；
           超时必须抛 :class:`ApiTimeoutError`，不允许静默返回旧状态；
        3. 错误码：设备错误必须映射为 :class:`HardwareError` 并保留
           ``code``/``detail``，不允许把供应商异常直接抛给上层；
        4. 遥测：返回的 dict 必须带 ``units``/``timestamp``/``settled``
           等字段，且可 JSON 序列化（管线 manifest 会原样记录）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: 本管线内部的统一波长约定（与 products.optics / observability 一致）。
WAVELENGTH_NM = 632.8

#: 所有接口实现都应带上的 provenance 字段（manifest 直接透传）。
UNIT_SYSTEM = {
    "elevation_deg": "deg (仰角, 地平线以上, [0, 90])",
    "azimuth_deg": "deg (方位角, 正北起算、向东为正, [0, 360))",
    "ra_deg": "deg (J2000 赤经)",
    "dec_deg": "deg (J2000 赤纬)",
    "time": "UTC ISO-8601 (YYYY-MM-DDTHH:MM:SS.sssZ)",
    "exposure_s": "s (单帧曝光时间)",
    "adu": "ADU (探测器原始计数, 含 bias/dark, 未做 flat/坏像元/宇宙线改正)",
    "zernike": "waves (632.8 nm 波前, 与 products.optics.forward 一致)",
    "mechanism_unit": "每个 DOF 在 dofs() 的 unit 字段中声明 (mm / mrad / deg / K)",
}


class AcquisitionError(RuntimeError):
    """数据采集层统一异常基类（adapter 必须把供应商错误映射到它）。

    Parameters
    ----------
    message:
        人类可读描述（中文/英文都可以）。
    code:
        稳定字符串错误码，例如 ``"TIMEOUT"`` / ``"OUT_OF_RANGE"`` /
        ``"NOT_CONNECTED"`` / ``"DEVICE_FAULT"``。上层据此决定重试策略。
    detail:
        可选的结构化细节（供应商原始错误号、寄存器值等），必须可 JSON
        序列化。
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class ApiTimeoutError(AcquisitionError):
    """运动/采集在 timeout 内没有完成（adapter 必须抛出，不能静默返回）。"""


class HardwareError(AcquisitionError):
    """设备故障/通讯失败（adapter 必须保留供应商错误码到 ``code``）。"""


class TelescopeTerminalApi(ABC):
    """望远镜控制终端 API 的抽象边界。

    工作流位置：俯仰角扫描（15°/45°/75° …）时由管线调用，用于指向目标
    天区并取回工况/遥测。

    ``no vendor API is available yet; implement this class in an adapter
    module when the terminal/camera/mechanism SDK arrives``

    Adapter 必须保证
    ----------------
    单位（units-of-record）:
        * ``elevation_deg``：度，地平线以上，闭区间 ``[0, 90]``（超出即
          ``ValueError``，不要静默截断）；
        * ``azimuth_deg``：度，正北起算、向东为正，``[0, 360)``；
        * ``time``：UTC ISO-8601；``ra_deg``/``dec_deg``：J2000 度；
        * 角速度/时间：度/秒、秒。
    时序:
        * ``goto_elevation``/``goto_azimuth`` 必须阻塞到“编码器到位且
          settle 完成”才返回；超过 ``timeout_s`` 抛 :class:`ApiTimeoutError`；
        * 返回的 ``status()`` 必须带 ``settled=True`` 和实际 ``timestamp``。
    错误码:
        * 超时 ``TIMEOUT``；目标越限 ``OUT_OF_RANGE``；未连接
          ``NOT_CONNECTED``；设备故障 ``DEVICE_FAULT``。
    必需遥测（``status()`` 至少包含）:
        ``connected``, ``state``, ``elevation_deg``, ``azimuth_deg``,
        ``settled``, ``timestamp``, ``limits``, ``servo``（跟踪误差/速度）,
        ``operating_condition``（或显式声明为 ``None`` + 原因）。
    """

    #: provenance: "mock" 表示显式 mock；真实 adapter 必须用 "hardware"。
    mode: str = "hardware"

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """返回终端当前状态与遥测（必须 JSON 可序列化）。

        Returns
        -------
        dict
            ``connected`` / ``state`` / ``elevation_deg`` / ``azimuth_deg`` /
            ``settled`` / ``timestamp`` / ``limits`` / ``servo`` /
            ``operating_condition`` / ``units`` 等字段。其中
            ``operating_condition`` 若终端不提供工况数据，必须显式写
            ``None`` 并给出 ``"reason"``，不允许伪造。
        """

    @abstractmethod
    def goto_elevation(self, deg: float, timeout_s: Optional[float] = None
                       ) -> Dict[str, Any]:
        """俯仰角扫描：转到 ``deg``（度）并阻塞到稳定。

        Parameters
        ----------
        deg:
            目标仰角，度，``[0, 90]``；越限必须 ``ValueError``。
        timeout_s:
            阻塞超时（秒）；``None`` 表示用终端默认超时；超时抛
            :class:`ApiTimeoutError`。

        Returns
        -------
        dict
            完成后 ``status()`` 的快照（含实际仰角和稳定标志）。
        """

    @abstractmethod
    def goto_azimuth(self, deg: float, timeout_s: Optional[float] = None
                     ) -> Dict[str, Any]:
        """方位角：转到 ``deg``（度，正北起算向东为正，``[0, 360)``）。

        语义同 :meth:`goto_elevation`（阻塞到稳定、超时抛异常、单位度）。
        """

    @abstractmethod
    def pointing(self) -> Dict[str, Any]:
        """返回当前指向（不含运动）：仰角/方位/赤经赤纬 + 历元。

        Returns
        -------
        dict
            ``elevation_deg``, ``azimuth_deg``, ``ra_deg``, ``dec_deg``,
            ``epoch``（如 ``"J2000"``）, ``settled``, ``timestamp``, ``units``。
            真实 adapter 必须给出由编码器/星表解算的真实指向，不能返回目标
            指令值冒充实测值。
        """


class CameraApi(ABC):
    """天文相机终端的抽象边界（离焦焦面 CCD 拍摄）。

    工作流位置：每个补偿机构位姿拍摄一组离焦 PSF 帧；输出进入
    ``products.preprocess`` 做 bias/dark/flat、坏像元、背景、饱和、宇宙线、
    星点检测与配准。

    ``no vendor API is available yet; implement this class in an adapter
    module when the terminal/camera/mechanism SDK arrives``

    重要约定：``capture`` 返回的帧是 **raw ADU**——包含 bias/dark，未做
    flat、坏像元修复、背景扣除或宇宙线剔除；任何线性化/去马赛克/装箱都
    必须在 metadata 中声明（管线只会按声明的约定处理）。

    Adapter 必须保证
    ----------------
    单位:
        * 数组: ``float64/整数``，单位 ADU（``metadata()["adu_gain_e_per_adu"]``
          给出电子到 ADU 的增益，便于把噪声换算回电子）；
        * ``exposure_s``: 秒；``defocus_list``: 由管线给定的离焦量（本产品
          定义为“等效 Z4 波前波数”，与 products.optics.forward 同一约定）。
    时序:
        * ``capture`` 必须在一个完整读出周期后返回；曝光重叠/异步读出必须
          在 metadata 中声明，否则上层假定帧是按顺序的独立曝光；
        * 帧序必须与 ``defocus_list`` 顺序一一对应（``K = len(defocus_list)``）。
    错误码:
        * 曝光超时 ``TIMEOUT``；饱和未处理属正常数据（返回 ADU 上限）；
          相机断连 ``NOT_CONNECTED``；读出故障 ``DEVICE_FAULT``。
    必需 metadata（至少）:
        ``shape``, ``bit_depth``, ``adu_max``, ``adu_gain_e_per_adu``,
        ``read_noise_e``, ``readout_bias_adu``, ``dark_current_e_per_s``,
        ``exposure_s``, ``defocus``, ``units``, ``mode``, ``label``；
        可选 ``flat_field``/``defect_map``/``cr_exclusion`` 等相机配置。
    """

    mode: str = "hardware"

    @abstractmethod
    def capture(self, defocus_list: Sequence[float], exposure_s: float
                ) -> np.ndarray:
        """拍摄一组离焦帧，返回 **raw ADU** 数组 ``[K, N, N]``。

        Parameters
        ----------
        defocus_list:
            离焦量序列（等效 Z4 波前波数，长度 K>=1）。相机不必理解物理
            含义，只按序拍摄；顺序必须与返回帧一致。
        exposure_s:
            单帧曝光时间（秒），``>=0``；0 表示读出 bias 级（暗曝光）。

        Returns
        -------
        numpy.ndarray
            形状 ``[K, N, N]`` 的 raw ADU（含 bias/dark，未定标）。
        """

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """返回相机/探测器配置与最近一次拍摄信息（JSON 可序列化）。

        必须包含 :class:`CameraApi` 类文档列出的必需字段；``shape`` 用于
        预处理的 master frame 尺寸校验，``adu_max`` 用于饱和掩膜，增益/
        读出噪声用于误差预算。
        """


class MechanismApi(ABC):
    """补偿机构控制（M2 调焦/平移/倾斜等）的抽象边界。

    工作流位置：灵敏度矩阵标定（逐 DOF 施加 ±A 激励）与每个俯仰角下的
    补偿量求解。DOF 的单位由 :meth:`dofs` 声明，管线 **不做单位换算**，
    求解出的命令直接以该单位下发。

    ``no vendor API is available yet; implement this class in an adapter
    module when the terminal/camera/mechanism SDK arrives``

    Adapter 必须保证
    ----------------
    单位:
        * 每个 DOF 的 ``unit``（``mm``/``mrad``/``deg``/``K``…）与
          ``low``/``high`` 同单位；``step`` 是该 DOF 的最小可分辨步长；
        * ``readback`` 必须返回实测值（编码器/光栅），不是指令回显。
    时序:
        * ``move`` 阻塞到“到位 + settle”才返回；超过 ``timeout_s``（若
          adapter 支持）抛 :class:`ApiTimeoutError`；
        * ``home`` 之后 ``readback`` 必须为各 DOF 的 home 值（本产品约定
          为 0，但必须在 ``dofs``/metadata 里写明）。
    错误码:
        * 越界 ``OUT_OF_RANGE``（建议在 adapter 内直接 ``ValueError``）；
          回零失败/限位 ``HOMING_FAILED``；设备故障 ``DEVICE_FAULT``。
    安全:
        * 真实 adapter 必须实现软/硬限位、急停和上电回零检查；mock 只
          模拟限位与步长，不代表真实装置的安全行为。
    """

    mode: str = "hardware"

    @abstractmethod
    def dofs(self) -> List[Dict[str, Any]]:
        """返回可控 DOF 列表（顺序即灵敏度矩阵列顺序）。

        Returns
        -------
        list of dict
            每项至少 ``{"name": str, "unit": str, "low": float,
            "high": float, "step": float}``；可选 ``label``/``physical``/
            ``home``/``note``。``low <= home <= high``，``step > 0``。
        """

    @abstractmethod
    def move(self, name: str, value: float) -> None:
        """把 DOF ``name`` 移动到 ``value``（该 DOF 的 ``unit``），阻塞到稳定。

        越界必须 ``ValueError``；未知 DOF 必须 ``KeyError``；运动失败抛
        :class:`HardwareError`。adapter 可以在内部做量化/回差补偿，但必须
        在 ``dofs()``/metadata 中声明，否则上层会假定理想执行。
        """

    @abstractmethod
    def home(self) -> None:
        """所有 DOF 回零并等待稳定（失败抛 :class:`HardwareError`）。"""

    @abstractmethod
    def readback(self) -> Dict[str, float]:
        """返回各 DOF 的实测当前值 ``{name: value}``（同 ``dofs`` 单位）。"""
