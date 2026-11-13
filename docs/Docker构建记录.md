# Docker 构建记录

> 日期：2026-11-13  
> 机器：WSL2 / Ubuntu 20.04，Docker 26.1.3  
> 目标镜像：`telescope-mvp:test`

## 1. Dockerfile 内容

`Dockerfile`（仓库根目录）做三件事：

1. Ubuntu 20.04 + build-essential/CMake/Python3/numpy/scipy/astropy/matplotlib/Eigen/GTest/Benchmark/pybind11；
2. 构建 `numerics/`：GTest + Benchmark + pybind11；
3. 构建主项目 CPU 层（`TEL_BUILD_CUDA=OFF`）并跑 CTest（CPU + pipeline）。

## 2. 实际构建尝试

```bash
cd /home/legend-star/telescope_gpu
docker build -t telescope-mvp:test .
```

结果：

```text
Step 1/8 : FROM ubuntu:20.04
Get "https://registry-1.docker.io/v2/": net/http: request canceled
while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

本机网络无法访问 Docker Hub（`registry-1.docker.io` 超时），所以**没有产生成功镜像**。
这不是 Dockerfile 逻辑错误；同一提交的 GitHub Actions `cpu-and-pipeline` / `numerics`
两个 job 已经通过。

## 3. 在可访问 Docker Hub 的机器上复现

```bash
git clone git@github.com:Starry-Journey/telescope_gpu.git
cd telescope_gpu
git checkout engineering-mvp

docker build -t telescope-mvp:test .
docker run --rm telescope-mvp:test \
  ctest --test-dir build --output-on-failure
docker run --rm telescope-mvp:test \
  ctest --test-dir build-numerics --output-on-failure
```

预期：

```text
100% tests passed, 0 tests failed out of 3     # 主项目 CPU + pipeline
100% tests passed, 0 tests failed out of 10    # numerics GTest
```

如果 Docker Hub 仍慢，可换镜像源/代理，或先 `docker pull ubuntu:20.04` 再 build。
GPU 镜像不在本 Dockerfile 内；CUDA 需要 NVIDIA Container Toolkit + 基础镜像。

## 4. 结论

- Dockerfile 已提交，命令/依赖/验收标准明确；
- 本机因 Docker Hub 网络限制未完成构建；
- 不要把“未构建”写成“已验证”；在能联网的机器上补一次上面的构建记录即可。
