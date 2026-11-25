# WSL 开发环境（Ubuntu 20.04 + CUDA 12.4 + Python 3.12）

> 本文只描述**本机 WSL 开发环境**，不改变产品的运行契约。编译器/依赖版本、
> 为什么这样选、以及“哪些工具故意不装”都在这里说明，避免重复安装同类库。
>
> 一键脚本：[../scripts/setup_wsl_dev.sh](../scripts/setup_wsl_dev.sh)（幂等）。
> 代码审阅与 HPC/AI4S 差距见 [代码审阅-2026-11-22.md](代码审阅-2026-11-22.md)。

## 1. 本机事实（2026-11-22）

| 项 | 值 | 说明 |
|---|---|---|
| 发行版 | **Ubuntu 24.04.5 LTS（glibc 2.39）** | 2026-11-22 从 20.04 迁移完成，旧发行版已备份并注销 |
| CPU / 内存 | i5-13500H，16 线程 / 7.6 GB | WSL 默认内存上限，跑大网格注意 OOM |
| GPU | RTX 3050 Laptop 4 GB，sm_86 | `nvidia-smi` 由 `/usr/lib/wsl/lib` 提供 |
| 驱动 / CUDA | 驱动 581.80，toolkit 12.6（nvcc 12.6.85） | `nvcc` 在 `/usr/local/cuda-12.6/bin`，`/usr/local/cuda` 已指向它 |
| C++ | GCC 13.3（系统默认） | 不再装第二套编译器 |
| 构建 | CMake 3.28.3 + Ninja 1.11.1 + ccache 4.9.1 | 预设见根目录 `CMakePresets.json` |
| HPC 库 | OpenMPI 4.1.6、HDF5 1.10.10、OpenBLAS 0.3.26、FFTW 3.3.10、hwloc 2.10 | 一套 MPI / 一套 BLAS，不混装 MPICH/MKL |
| Python | `.venv` = CPython 3.12.3（uv 管理） | 系统 Python 也是 3.12.3；不要再装 conda |
| 关键包 | numpy 2.5.5、scipy 1.18.1、astropy 8.0.3、matplotlib 3.11.2、nlopt 2.11、PySide6 6.9.3、torch 2.6.0+cu124、h5py 3.16、mpi4py 4.1.2、pybind11 3.1 | `torch.cuda.is_available() == True` |

## 2. 一键准备

```bash
cd ~/telescope_engineering
bash scripts/setup_wsl_dev.sh            # 只装缺的包，重复执行安全
bash scripts/setup_wsl_dev.sh --verify   # 额外跑一遍 CPU 构建 + CTest + Python 测试
```

脚本只做三件事：

1. 用 `dpkg-query` 检查并按需 `apt-get install`（已装则跳过）；
2. 用 uv 建 `.venv`（CPython 3.12），依赖**只从 `pyproject.toml` 读取**；
3. 自检 Python 依赖、CUDA、`gcc/g++/cmake/ninja/ccache/gdb/mpicc/mpirun`。

`torch` 单独从 `https://download.pytorch.org/whl/cu124` 安装（PyPI 默认是 CPU 版）；
其余包走清华 PyPI 镜像。

> 全新 24.04 默认没有 `nvcc`。如果需要编译 `linux-cuda` 目标，先按 §8 安装
> CUDA toolkit（只装 toolkit，驱动由 Windows 提供），再运行本脚本；只做
> AI4S/torch 训练时可以先跳过。脚本发现没有 nvcc 会提示并退化为 CPU 构建。

## 3. 刻意不装的工具（去重清单）

| 已有 | 不再装 | 原因 |
|---|---|---|
| `uv` | conda / mamba / poetry / virtualenv / pipenv | 一个解释器/环境管理器足够；仓库已有 `uv` |
| GCC 9 | clang / clang++ / lldb | C++ 编译器只保留一套；`clangd` 也会和已装的 C/C++ 扩展重复 |
| `clang-format-12` / `clang-tidy-12` | 第二套格式化/静态分析 | 只装工具二进制，不装 clang 编译器 |
| `ruff`（仓库已配置） | flake8 / black / isort | 一个 linter+formatter 覆盖 E/F/I |
| OpenMPI | MPICH / Intel MPI | 只保留一套 MPI ABI |
| OpenBLAS | MKL / ATLAS / 参考 LAPACK | `libopenblas-dev` 已通过 alternatives 提供 BLAS/LAPACK |
| 系统 HDF5（串行） | `libhdf5-openmpi-dev` | 两者互斥；单节点先用串行 HDF5，上集群再换并行版 |
| torch + CUDA | JAX / TensorFlow / CuPy | 现有 GPU 栈只有一套；避免第二套 CUDA 运行时/显存池 |
| 已装 C/C++、CMake Tools、Python 扩展 | `cpptools-extension-pack`、clangd 扩展、Makefile Tools | 功能重叠，见 `.vscode/extensions.json` |
| `pip` 版 pybind11 3.x | apt 的 `pybind11-dev` 2.4 | 2.4 不支持 CPython 3.12 |
| 系统 GTest（`/usr/lib`） | 每个 build 目录再 FetchContent 一份 | 见 §5 的 WSL 污染处理 |
| CUDA toolkit 自带 `nsys`/`ncu` | Nsight VS Code 扩展、nvprof 额外包 | 命令行 profiler 已够用 |

不装 `pandas`/`h5py` 之外的表格库、不装 SEP/photutils/OpenCV：`pipeline/`
用 `numpy + scipy.ndimage` 已实现同等功能；等密集星场真的需要 PSF 拟合时再评估。

## 4. 构建与测试（CMakePresets）

```bash
# CPU：telescope_cpu + numerics(Eigen/GTest/Benchmark/pybind11) + Python 管线测试
cmake --preset linux-cpu
cmake --build --preset linux-cpu
ctest --preset linux-cpu --output-on-failure

# CUDA：正向模型 + 相位差求解器 + GPU 测试（RTX 3050 = 86）
cmake --preset linux-cuda
cmake --build --preset linux-cuda        # preset 里 jobs=2，避免 nvcc OOM
ctest --preset linux-cuda --output-on-failure

# framework 垂直切片
cmake --preset linux-framework
cmake --build --preset linux-framework
ctest --preset linux-framework --output-on-failure
```

预设统一做了这些事（不再在文档/任务里重复写）：

- Ninja + ccache（C/C++/CUDA 都走 ccache）；
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON`，供 C/C++ 扩展和 clang-tidy 使用；
- `TEL_PYTHON_EXECUTABLE` / `Python_EXECUTABLE` 指向 .venv/bin/python，
  避免 pybind11 找到系统 3.8；
- `TEL_NUMERICS_USE_SYSTEM_GTEST=ON`，用 `/usr/lib` 的 GTest，不重复下载；
- `CMAKE_IGNORE_PATH` / `CMAKE_IGNORE_PREFIX_PATH` 忽略 `/mnt/c`…`/mnt/g`，
  防止 CMake 把 Windows conda 的库当成 Linux 依赖。

换 GPU（A100=80、H100=90 等）时覆盖架构即可：

```bash
cmake --preset linux-cuda -DCMAKE_CUDA_ARCHITECTURES=80
```

Python 侧：

```bash
.venv/bin/python -m unittest discover -s products/tests -v
.venv/bin/python -m unittest discover -s pipeline/tests -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s desktop/tests -v
QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/e2e_smoke.py
```

> 当前 `.venv` 里是 PySide6 6.9.3，而不是 Windows 打包用的 6.12.0：
> 6.10+ 的 Linux wheel 要求 glibc ≥ 2.34，Ubuntu 20.04 只有 2.31。
> `pyproject.toml` 已按 `sys_platform` 分开固定，所以 Windows 打包不受影响。

## 5. VS Code（WSL Remote）

1. Windows 侧安装 `ms-vscode-remote.remote-wsl`（已加进推荐列表）；
2. 在 WSL 里执行 `code .`，左下角显示 `WSL: Ubuntu-20.04`；
3. 首次打开会提示安装 `.vscode/extensions.json` 里的推荐扩展；
4. 状态栏选择 CMake preset（`linux-cpu` / `linux-cuda`），
   `Tasks: Run Task` 里有 configure/build/test/lint/e2e；
5. 调试：`launch.json` 提供 `telescope_tests`、`numerics_tests`、
   `framework_demo` 和当前 Python 文件（用 .venv/bin/python）。

> 本机 VS Code 已有 C/C++、CMake Tools、Python/Pylance/debugpy 等扩展；
> WSL 侧已通过 Open VSX 离线补装 `charliermarsh.ruff` 与
> `tamasfe.even-better-toml`。宿主机侧的 `ms-vscode-remote.remote-wsl`
> 因 marketplace 超时未装成功，需在扩展面板手动安装或配置代理后重试。
> 不需要再装 `cpptools-extension-pack`、clangd 扩展或 Makefile Tools。

## 6. 已知边界

1. **已迁到 Ubuntu 24.04（glibc 2.39）**：PySide6 6.12 已可用；当前
   `pyproject.toml` 的 Linux 分支仍固定 6.9.3（与 20.04 兼容），需要时可单独升级。
2. **系统 Python 现在是 3.12.3**：项目统一用 .venv/bin/python，不要
   `sudo pip` 装包；旧 3.8 基线只作回归参考。
3. **4 GB 显存**：CNN 训练/批量 PD 要控制 batch/网格；256² 多起点更适合
   `scripts/benchmark_pd.py` 的离线流程，不要放进 GUI 同步调用。
4. **MPI 只在单节点**：WSL 的 OpenMPI 可以跑多 rank，但没有 InfiniBand/
   Slurm；真正的集群调度需要另配（见审阅文档的 HPC 路线）。
5. **WSL 的 PATH 会带 Windows 目录**：`/mnt/c/.../miniconda3` 在最前面时，
   CMake 可能搜到 Windows 的 `.lib`/config。预设已加忽略规则；手工构建时
   请显式传 `-DTEL_NUMERICS_USE_SYSTEM_GTEST=ON` 并检查 `GTest` 路径。

## 7. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `Could NOT find PythonInterp ... 3.8.10` | pybind11 找到了系统旧 Python（迁移前 3.8） | 用 preset；或传 `-DPython_EXECUTABLE=.venv/bin/python -DPYBIND11_FINDPYTHON=ON` |
| `GTest ... /mnt/c/.../miniconda3` | CMake config 模式搜到 Windows conda | 用 preset；或 `-DTEL_NUMERICS_USE_SYSTEM_GTEST=ON`（脚本已限定 `/usr/lib`） |
| `pyside6 has no wheels with a matching platform tag` | 这是 20.04/glibc 2.31 的历史问题；24.04 已可装 6.10+ | 24.04 无需处理；项目 Linux 分支仍固定 6.9.3 |
| `nvcc: command not found` | CUDA 没进 PATH | `source ~/.bashrc`；脚本会追加 `/usr/local/cuda-12.6/bin`（或 `/usr/local/cuda/bin`） |
| `torch.cuda.is_available() == False` | 装成了 CPU 版 torch | 重装：`uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0` |
| `mpirun` 卡住/报 root | WSL 默认 TCP btl 或 root 用户 | 用普通用户运行；加 `--mca btl self,vader`（或 `export OMPI_MCA_btl=self,vader`）；root 需 `--allow-run-as-root` |

## 8. 从 Ubuntu 20.04 迁移到 24.04（已完成）

> 本机已于 2026-11-22 完成迁移：旧 20.04 已完整导出备份后注销，24.04 全量
> 构建/测试通过。以下步骤保留给以后迁机或重装参考。

**不要复制 `.venv`。** 它是 3.12 + glibc 2.31 的本地环境（本机约 5.9 GB），
复制到 24.04 既慢又不可用；`ruff`/`torch` 这类可执行文件在 Explorer 跨
WSL 的 9p 共享路径复制时还容易报“无法读源文件或磁盘”。迁移只带源码和数据。

在 **20.04** 里打包（排除可重建物）：

```bash
cd ~
tar --exclude='telescope_engineering/.venv' \
    --exclude='telescope_engineering/build*' \
    --exclude='telescope_engineering/.pytest_cache' \
    --exclude='*/__pycache__' \
    -czf /mnt/c/Users/17424/Downloads/te-src.tgz telescope_engineering
```

在 **24.04** 里解包并修正属主：

```bash
cd ~
tar -xzf /mnt/c/Users/17424/Downloads/te-src.tgz
sudo chown -R "$USER:$USER" ~/telescope_engineering
sudo chown -R "$USER:$USER" ~        # 修掉迁移产生的 root 属主（.aider/.ansible 等）
rm -rf ~/venv          # 迁移过程中产生的半成品 venv，改用项目 .venv
```

装 CUDA toolkit（WSL 只装 toolkit，GPU 驱动由 Windows 提供；nvidia-smi 已可用）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6   # 或更新版本；CMake 会找 /usr/local/cuda
```

然后重建环境：

```bash
cd ~/telescope_engineering
bash scripts/setup_wsl_dev.sh --verify
```

> 24.04 的 glibc 是 2.39，PySide6 6.12 也能装；但 `pyproject.toml` 目前
> Linux 仍固定 6.9.3（20.04 兼容），需要时再单独升级。

