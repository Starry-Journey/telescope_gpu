#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace telescope {

inline void throw_cuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) +
                             " at " + file + ":" + std::to_string(line) + " (" + expr + ")");
}

inline void throw_cufft(cufftResult err, const char* expr, const char* file, int line) {
    if (err == CUFFT_SUCCESS) return;
    throw std::runtime_error(std::string("cuFFT error: ") + std::to_string(err) +
                             " at " + file + ":" + std::to_string(line) + " (" + expr + ")");
}

#define TEL_CUDA_CHECK(call) \
    ::telescope::throw_cuda((call), #call, __FILE__, __LINE__)

#define TEL_CUFFT_CHECK(call) \
    ::telescope::throw_cufft((call), #call, __FILE__, __LINE__)

inline int div_up(int a, int b) { return (a + b - 1) / b; }

// 极简 RAII 设备缓冲区。禁止拷贝，允许移动。
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);  // 析构中不抛异常
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        if (count == 0) return;
        TEL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        count_ = count;
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() const { return ptr_; }
    T** addr() { return &ptr_; }
    std::size_t count() const { return count_; }
    void memset_zero() { TEL_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace telescope
