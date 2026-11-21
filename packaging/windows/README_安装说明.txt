TelescopeEngineering · 大口径望远镜重力变形标定 · 波前反演数据链路 v1.0.0
================================================================

【启动】双击 TelescopeEngineeringQtNative.exe（或桌面/开始菜单快捷方式）。
      6 个页签：CCD/PSF 仿真 · 相位差反演 · FEA · ZMX 布局 · S/控制 · 演示。
      长任务经 worker 子进程执行，可取消；结果与图片可从各页签导出。

【本包已内置】真实 Zemax 3x5 灵敏度矩阵（EXTERNAL-CHECK）、
Abaqus M1 工况数据（FEA-PROVISIONAL）、全部服务层代码与运行时（无需 Python）。

【可选外部依赖（不影响启动）】
- CUDA 反演引擎：需本机 NVIDIA GPU + 单独编译的 inverse_solve（源码包内 cmake 构建）
- Zemax 重算：需本机 OpticStudio ZOS-API 许可
- Abaqus 重算：需本机 Abaqus 2025+ 许可
未安装以上工具时，相关页签使用内置数据/mock，界面功能不受影响。

【卸载】Windows 设置 → 应用 → TelescopeEngineering → 卸载
（或控制面板；仅删除本程序目录与快捷方式，不写注册表外的系统位置）。

【数据标签】CORE-VERIFIED / SYNTHETIC-VALIDATED / EXTERNAL-CHECK /
FEA-PROVISIONAL / UNVERIFIED —— 引用任何数值请连同标签一起引用。
