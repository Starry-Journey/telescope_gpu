# ZOS-API 与 Abaqus 实测记录

> 机器：`STAR-DREAM`，2026-11-16
> 目的：确认能否直接生成真实 3A S 和 M1 简化 FEA，而不是继续用合成数据。

## 1. Zemax / OpticStudio

安装：

```text
D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00\OpticStudio.exe
D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00\ZOSAPI.dll
D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00\ZOSAPI_Interfaces.dll
```

### 1.1 `CreateNewApplication()`

结果：

```text
app = ZemaxUI.Common.ViewModels.ZOSAPI_Application
NumberOfOpticalSystems = 0
```

随后 `LoadNewSystem("2.5m主系统.ZMX")` 在非交互会话里卡住/超时。
没有生成 `sensitivity.csv`。

### 1.2 `CreateZemaxServer(...)`

分别传 install dir / `OpticStudio` / `OpticStudio.exe`，结果：

```text
Mode=Server
LicenseStatus=Unknown
IsValidLicenseForAPI=False
InitializationErrorCode=Initialization failed
InitializationErrors=App path: D:\Program Files\...\OpticStudio
System.ComponentModel.Win32Exception
The system cannot find the file specified
```

结论：**当前会话无法完成 headless ZOS-API 加载**，因此没有真实 3×5 S。
`ZMX` 结构解析仍可用；真实 S 必须等 OpticStudio 在可交互/可授权会话中跑通。

脚本：`scripts/zos_probe.ps1`

```powershell
powershell -ExecutionPolicy Bypass -File scripts\zos_probe.ps1 `
  -InstallDir "D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00" `
  -Zmx "C:\temp\model.zmx" -Log "C:\temp\zos_probe.log"
```

退出码：0 成功解析模型；2 连接/许可失败；3 LoadNewSystem 失败。

## 2. Abaqus

安装：

```text
D:\SIMULIA\Commands\abq2025.bat
D:\SIMULIA\EstProducts\2025\win_b64
```

`abq2025 information=version` 能返回：

```text
Abaqus 2025
Internal Version: Abaqus 6.25-1
```

但 `cae noGUI` 失败：

```text
Cannot connect to license server system.
Feature: cae
Server name: 127.0.0.1
License path: 27890@localhost;27011@STAR-DREAM;
FlexNet Licensing error:-15,10  WinSock: Connection refused
```

结论：**Abaqus license server 没有运行/未连上**，没有生成 ODB 或面形 CSV。
没有启动或修改 license 文件，也没有绕过许可。

## 3. 简化主镜模型脚本

`scripts/abaqus_m1_simple.py` 已写好，但**未实际求解**：

- 环形 Zerodur 盘：OD 2500 mm、ID 700 mm、厚 260 mm；
- 36 底支撑（6-12-12-6）+ 24 侧支撑；
- 重力 0°/90°；
- 输出顶面 `x,y,z,u1,u2,dz_nm` CSV。

运行（license 可用时）：

```bat
cd C:\temp\abaqus_m1
"D:\SIMULIA\Commands\abq2025.bat" cae noGUI=C:\temp\abaqus_m1\abaqus_m1_simple.py
```

预期输出：

```text
C:\temp\abaqus_m1\M1_0deg_top.csv
C:\temp\abaqus_m1\M1_90deg_top.csv
```

这是**简化板模型**，不是完整 2.5 m 整机；支撑/接触/材料需收敛验证后才能替换
`data/fea_cases/M1_*.csv`。

## 4. 数据状态

| 数据 | 状态 |
|---|---|
| ZMX 结构 | 已解析：2500 mm、8 面、stop=2 |
| 真实 Zemax S | **未生成**（ZOS-API 不可用） |
| Abaqus 真实 ODB | **未生成**（license 不可用） |
| 简化 FEA 脚本 | 已就绪，未运行 |
| 真实多离焦 FITS | 没有 |
| 合成 FEA/PD | 已有，标注 synthetic |

## 5. 下一步

1. 在可交互 Windows 会话中启动 OpticStudio，确认 `IsValidLicenseForAPI=True`；
   再跑 `zos_probe.ps1`，能加载后接 `sensitivity.csv` 生成；
2. 启动合法 Abaqus license server，或把 `ABAQUSLM_LICENSE_FILE` 指向可用服务器；
3. 先用简化模型跑通 0°/90°，检查位移量级和支撑反力；
4. 再把 ODB 顶面 CSV 上传 `/api/fea/upload`，生成 66 项 Zernike；
5. 没有真实多离焦前，PD 仍只做合成验证。
