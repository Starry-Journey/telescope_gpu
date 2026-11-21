# framework/ — 光学仪器反演与控制框架

把现有相位差求解器/CCD 管线/数值库收敛到同一组契约：

```text
Frame/Observation -> ParameterSpace -> ForwardModel -> InverseSolver
                                      -> SensitivityBuilder -> ControlSolver
```

## 构建

```bash
cmake -S . -B build-framework \
  -DTEL_BUILD_CUDA=OFF \
  -DTEL_BUILD_FRAMEWORK=ON \
  -DTEL_BUILD_NUMERICS=ON \
  -DTEL_NUMERICS_BUILD_TESTS=OFF \
  -DTEL_NUMERICS_BUILD_BENCH=OFF \
  -DTEL_NUMERICS_BUILD_PYTHON=OFF
cmake --build build-framework -j4
ctest --test-dir build-framework -R framework --output-on-failure
./build-framework/framework/framework_demo results/framework
```

## 目录

```text
include/instrument/
  data.hpp          Image/Frame/Observation/Status
  parameter.hpp     ParameterSpace/ControlSpace
  model.hpp         ForwardModel/NoiseModel/TargetExtractor
  inverse.hpp       InverseSolver/Solution
  sensitivity.hpp   SensitivityMatrix/FiniteDifferenceSensitivity
  control.hpp       ControlSolver/WeightedLeastSquaresControl
  registry.hpp      Registry<T>
  provenance.hpp    RunManifest
src/                parameter/sensitivity/control/provenance
plugins/
  synthetic_linear  z = A u + z0（测试用）
  phase_diversity   包装现有 CUDA PhaseDiversitySolver（TEL_BUILD_CUDA=ON）
apps/framework_demo.cpp
tests/test_framework.cpp
```

## 结果

- 3×5 灵敏度矩阵：rank=3，condition=1.0897，nullspace=2；
- 控制目标残差 < ~1e-9；
- 5×5 满秩控制恢复 < 1e-6；
- 输出 `sensitivity.csv`、`control_result.json`、`run_manifest.json`。

## 状态

v1 是可运行垂直切片，不是最终通用平台。下一步：pipeline→FITS bridge、JSON/YAML 配置/缓存、不确定度、QP、Zemax/poppy PSF。
