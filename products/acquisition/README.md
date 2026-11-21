# products/acquisition —— 厂商接口边界 + 显式 mock

> 端到端数据管线的三个外部边界：望远镜控制终端 / 天文相机 / 补偿机构。
> **今天没有真实供应商 API，也没有工况数据**：接口是契约，mock 是唯一可运行实现。

## 职责

- 固定接口语义（方法名、units-of-record、时序、错误码），真实 SDK 到位后只写 adapter；
- 接口层不 import 任何供应商 SDK、不发起网络/串口/PLC 调用；
- 提供显式 mock（`mode="mock"`、`label="SYNTHETIC-VALIDATED"`）把
  采集 → 预处理 → PD → 实测 S → 俯仰扫描补偿 → FEA 请求整链跑通。

## 输入 / 输出

- 接口层无输入；mock 由构造参数（seed、真值矩阵、运动学参数）驱动；
- 输出：`CameraApi.capture()` → raw ADU `[K,N,N]`（含 bias/dark，未定标）；
  `TelescopeTerminalApi.status()/pointing()` → 指向/遥测 dict；
  `MechanismApi.dofs()/readback()` → 各 DOF 位置（mm/mrad/deg/K）。

## 关键文件与函数

| 文件 | 关键接口 |
|---|---|
| `products/acquisition/interfaces.py` | `TelescopeTerminalApi`（status/pointing/goto_elevation/goto_azimuth）、`CameraApi`（capture/metadata）、`MechanismApi`（dofs/move/home/readback）、`AcquisitionError`/`ApiTimeoutError`/`HardwareError`、`UNIT_SYSTEM`、`WAVELENGTH_NM` |
| `products/acquisition/mock.py` | `MockTelescopeTerminal`、`MockCamera`、`MockMechanism`、`MockRig`、`build_mock_rig`、`MOCK_LABEL`/`MOCK_MODE` |

## 依赖

- 仅 numpy；mock 内部复用 `products.ccd` 的 PSF 与 `products.control` 的代理 S 作真值；
- 不依赖 Qt、网络、供应商 SDK。

## 标签与验证状态

- 接口：契约（无数据标签）；mock：SYNTHETIC-VALIDATED + `mode=mock`；
- 真实望远镜/相机/机构数据：**UNVERIFIED（无真机）**；adapter 落地前不得当实测。

## 常见坑

1. 不要为“跑通”改接口语义或让 mock 冒充 hardware；adapter 才是真实实现；
2. 单位以 `UNIT_SYSTEM` 为准（deg/mm/mrad/ADU/s），adapter 内部换算，别把供应商计数漏给管线；
3. `move`/`goto_*` 必须阻塞到“到位 + settle”，超时抛 `ApiTimeoutError`，不能静默返回旧状态；
4. mock 相机只支持 3 项 PD 状态（Z4/Z8/Z9）；扩展项会显式报错，不是静默忽略；
5. mock 返回的是星点窗口 ROI，不是整幅探测器；真实相机若返回整幅需在 metadata 声明裁剪。
