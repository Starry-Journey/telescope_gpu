#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WSL/Linux 开发环境一键准备（幂等，可重复执行）
#
#   bash scripts/setup_wsl_dev.sh            # 安装系统依赖 + 建 .venv
#   bash scripts/setup_wsl_dev.sh --verify   # 额外做一次 CPU 构建 + 测试
#
# 设计原则（与仓库其它文件保持单一来源，不重复引入同类工具）：
#   * Python 依赖只从 pyproject.toml 读取，不新增 requirements-*.txt；
#   * 只用 uv 管理解释器/虚拟环境，不再装 conda/poetry/virtualenv；
#   * 只用一套 MPI（OpenMPI）、一套 BLAS/LAPACK（OpenBLAS）、一套 HDF5；
#   * C++ 编译器沿用系统 GCC 9，clang 只装 clang-format/clang-tidy 工具，
#     不引入第二套编译器；
#   * lint 用 ruff（仓库已配置），不装 flake8/black/isort；
#   * Python 3.12 只用于 .venv；系统 3.8 保留但不再安装新包。
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY=0
[ "${1:-}" = "--verify" ] && VERIFY=1

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. 系统依赖（按 CMakeLists / CI 实际需要的包名安装；已装则跳过）
#    兼容 Ubuntu 20.04 与 24.04：clang 工具按仓库可用的版本选择。
# ---------------------------------------------------------------------------
if dpkg-query -W -f='${Status}' clang-format-12 2>/dev/null | grep -q 'install ok installed'; then
  CLANG_TOOLS=(clang-format-12 clang-tidy-12)
elif dpkg-query -W -f='${Status}' clang-format 2>/dev/null | grep -q 'install ok installed'; then
  CLANG_TOOLS=(clang-format clang-tidy)
elif apt-cache show clang-format-12 >/dev/null 2>&1; then
  CLANG_TOOLS=(clang-format-12 clang-tidy-12)
else
  CLANG_TOOLS=(clang-format clang-tidy)
fi

APT_PACKAGES=(
  # 构建工具（24.04 干净安装可能没有 gcc/make/cmake）
  build-essential cmake ninja-build ccache pkg-config
  # 调试与格式化（不装第二套编译器）
  gdb valgrind "${CLANG_TOOLS[@]}"
  # C++ 数学/测试依赖（CMake 的 find_package 优先于 FetchContent）
  libeigen3-dev libgtest-dev libbenchmark-dev
  # HPC：MPI + HDF5 + BLAS/LAPACK + FFT/NLopt + 拓扑
  libopenmpi-dev openmpi-bin libhdf5-dev hdf5-tools
  libopenblas-dev libhwloc-dev hwloc libnlopt-cxx-dev libfftw3-dev
  # Python 头文件（pybind11 / 系统构建）
  python3-dev
  # CLI 开发辅助
  curl ca-certificates git-lfs jq ripgrep fd-find
  # PySide6 无头运行所需的最小系统库
  libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1 libxrender1 libxi6 libxtst6
)

missing=()
for p in "${APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed'; then
    missing+=("$p")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  log "安装缺失的系统包：${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
else
  log "系统包已齐全，跳过 apt。"
fi

# ---------------------------------------------------------------------------
# 2. CUDA 进 PATH（只在缺失时追加，避免重复 export）
# ---------------------------------------------------------------------------
CUDA_HOME_GUESS=""
for d in /usr/local/cuda /usr/local/cuda-*; do
  [ -x "$d/bin/nvcc" ] && CUDA_HOME_GUESS="$d"
done

if [ -n "$CUDA_HOME_GUESS" ]; then
  if ! grep -q 'CUDA toolkit (added by' "$HOME/.bashrc" 2>/dev/null; then
    {
      echo ''
      echo '# CUDA toolkit (added by telescope_engineering/scripts/setup_wsl_dev.sh)'
      echo "export PATH=$CUDA_HOME_GUESS/bin:\$PATH"
      echo "export LD_LIBRARY_PATH=$CUDA_HOME_GUESS/lib64:\$LD_LIBRARY_PATH"
    } >> "$HOME/.bashrc"
    log "已把 $CUDA_HOME_GUESS/bin 写入 ~/.bashrc"
  fi
else
  warn "未找到 nvcc：CUDA 目标会退化为 CPU-only 构建。"
fi

# ---------------------------------------------------------------------------
# 3. uv + Python 3.12 虚拟环境
# ---------------------------------------------------------------------------
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  warn "未找到 uv，改用官方安装脚本（仅安装到 ~/.local/bin）"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
log "uv: $($UV --version)"

if [ ! -x .venv/bin/python ]; then
  log "创建 .venv (CPython 3.12)"
  "$UV" venv --python 3.12 .venv
fi
VENV_PY="$ROOT/.venv/bin/python"
log "venv: $($VENV_PY --version)"

# ---------------------------------------------------------------------------
# 4. 从 pyproject.toml 读取依赖（单一来源），torch 单独用 CUDA 索引安装
# ---------------------------------------------------------------------------
mapfile -t DEPS < <("$VENV_PY" - <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
deps = list(data["project"].get("dependencies", []))
for extra in ("ml", "hpc", "dev"):
    deps += list(data["project"].get("optional-dependencies", {}).get(extra, []))

# torch 由下面的 CUDA 索引单独安装；这里只保留非 torch 依赖。
# 注意：不能按包名去重——同一包可能出现带不同环境标记的条目
# （例如 Windows 用 PySide6 6.12、Linux 20.04 用 6.9.3），
# 按名字去重会把平台分支误删。
seen, out = set(), []
for dep in deps:
    name = dep.split(";")[0].split("=")[0].split(">")[0].split("<")[0].split("~")[0]
    name = name.strip().lower()
    if name == "torch":
        continue
    if dep in seen:
        continue
    seen.add(dep)
    out.append(dep)
print("\n".join(out))
PY
)

log "安装 CUDA 12.4 版 PyTorch（RTX 3050 = sm_86，4 GB 显存）"
"$UV" pip install --python "$VENV_PY" \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"

log "安装 pyproject 运行/开发依赖（共 ${#DEPS[@]} 条，已跳过 torch 重复项）"
# mpi4py 需要 MPI 编译器；显式指定，避免 uv 去猜。
MPICC="${MPICC:-/usr/bin/mpicc}" \
  "$UV" pip install --python "$VENV_PY" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "${DEPS[@]}"

# ---------------------------------------------------------------------------
# 5. 自检
# ---------------------------------------------------------------------------
log "自检：Python 依赖"
"$VENV_PY" - <<'PY'
import importlib
mods = ["numpy", "scipy", "astropy", "matplotlib", "nlopt", "h5py",
        "mpi4py", "PySide6", "pybind11", "torch", "pytest", "ruff"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001 - 自检需要打印所有失败
        print(f"  {m:12s} MISSING: {exc}")
        raise
import torch
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device = {torch.cuda.get_device_name(0)}")
PY

log "自检：C++ 工具链"
for t in gcc g++ cmake ninja ccache gdb mpicc mpirun; do
  printf '  %-8s %s\n' "$t" "$(command -v "$t" || echo MISSING)"
done

if [ "$VERIFY" = "1" ]; then
  log "配置并构建 CPU 目标（Ninja + ccache）"
  cmake --preset linux-cpu
  cmake --build --preset linux-cpu
  log "运行 CTest"
  ctest --preset linux-cpu --output-on-failure
  log "运行 Python 测试"
  "$VENV_PY" -m unittest discover -s products/tests -q
  "$VENV_PY" -m unittest discover -s pipeline/tests -q
fi

cat <<'EOF'

完成。后续用法：
  source .venv/bin/activate                 # 或直接用 .venv/bin/python
  cmake --preset linux-cuda && cmake --build --preset linux-cuda
  ctest --preset linux-cuda --output-on-failure
  QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/e2e_smoke.py

注意：WSL 里 /mnt/c 下的 Windows conda/Python 不要加进 PATH 前面；
     本仓库统一用 .venv/bin/python，避免与系统 3.8 的 site-packages 混用。
EOF
