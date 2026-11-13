# CPU-only MVP image: build/test the C++ numerics library and the CCD pipeline.
# CUDA is intentionally not included; use an NGC/PyTorch image for GPU work.
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        python3 python3-numpy python3-scipy python3-astropy python3-matplotlib \
        libeigen3-dev libgtest-dev libbenchmark-dev pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . .

# C++ numerical core: GTest + Benchmark + pybind11
RUN cmake -S numerics -B build-numerics -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build-numerics -j"$(nproc)" \
    && ctest --test-dir build-numerics --output-on-failure

# Main CPU layer + pipeline tests (CUDA disabled)
RUN cmake -S . -B build -DTEL_BUILD_CUDA=OFF -DTEL_BUILD_NUMERICS=OFF \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

CMD ["bash"]


