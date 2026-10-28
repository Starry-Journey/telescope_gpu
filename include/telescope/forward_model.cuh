#pragma once

#include <cufft.h>

#include <map>

#include "telescope/cuda_utils.cuh"
#include "telescope/types.hpp"

namespace telescope {

// GPU 正向模型：Zernike 波前 -> 光瞳复振幅 -> FFT -> PSF -> 降采样/归一化
// -> 多星点叠加。
//
// 设计要点（v2）：
//   * Zernike 基函数只在 CPU 端计算一次（make_pupil_grid），再上传 GPU。
//     CPU reference 与 GPU kernel 使用同一份基函数，避免两套公式漂移。
//   * 构造时一次性分配 workspace；热路径不再 cudaMalloc/cudaFree。
//   * 多帧离焦用 cuFFT planMany 一次批量 FFT。
//   * PSF 归一化在 GPU 上做（block 归约），不产生 D2H 同步。
//
// 约定：
//   * img_size = 256, oversample = 2 -> fft_size = 512。
//   * compute_psf 返回的 PSF 已归一化到 sum = 1。
//   * generate_frames 返回的星场图**未**归一化；代价函数比较前调用
//     normalize_frames（按总能量归一化）。
class ForwardModelGPU {
public:
    explicit ForwardModelGPU(const OpticalParams& params, int max_frames = 5);
    ~ForwardModelGPU();

    ForwardModelGPU(const ForwardModelGPU&) = delete;
    ForwardModelGPU& operator=(const ForwardModelGPU&) = delete;

    const OpticalParams& params() const { return params_; }

    // 上传星点表。重复调用会覆盖旧表；后续 generate_* 都使用这份表。
    void set_stars(const StarField& stars);

    // 单帧 PSF（sum = 1），写入调用方预分配的 device 缓冲。
    void compute_psf(const ZernikeCoeffs& coeffs,
                     float defocus,
                     float* d_psf_out);

    // 单帧星场图（不含背景/噪声，未归一化）。
    void generate_image(const ZernikeCoeffs& coeffs,
                        float defocus,
                        float* d_image_out);

    // 多帧星场图（不含背景/噪声，未归一化）。
    // h_defocus 长度 n_frames，n_frames <= max_frames。
    // 输出 d_frames_out 布局为 [frame][img_size * img_size]。
    void generate_frames(const ZernikeCoeffs& coeffs,
                         const float* h_defocus,
                         int n_frames,
                         float* d_frames_out);

    // 对每个 frame 做 sum 归一化（原地）。代价函数比较前调用。
    void normalize_frames(float* d_frames, int n_frames);

private:
    // 计算 n_frames 个 PSF（每个 sum = 1），结果放在 d_psf_down_。
    void run_psf_frames(const ZernikeCoeffs& coeffs,
                        const float* h_defocus,
                        int n_frames);

    // 返回 batch = n_frames 的 cuFFT plan；首次使用时创建并缓存。
    // v2.1 修正：原实现固定 batch=max_frames，n_frames<max 时会多算无用的
    // 帧（读未初始化显存、浪费 FFT 时间）。缓存后每个实际 batch 只建一次。
    cufftHandle batch_plan(int n_frames);

    OpticalParams params_;
    int max_frames_;

    cufftHandle plan_single_ = 0;
    std::map<int, cufftHandle> batch_plans_;

    // 光瞳采样：mask + 8 项 Zernike 基函数 [kNumZernike][fft_total]。
    DeviceBuffer<unsigned char> d_mask_;
    DeviceBuffer<float> d_basis_;

    // [max_frames][fft_total]
    DeviceBuffer<cufftComplex> d_pupil_;
    DeviceBuffer<cufftComplex> d_fft_;
    DeviceBuffer<float> d_psf_full_;
    // [max_frames][img_total]
    DeviceBuffer<float> d_psf_down_;

    DeviceBuffer<float> d_coeffs_;
    DeviceBuffer<float> d_defocus_;
    DeviceBuffer<float> d_sums_;

    // 星点坐标在上传时一次性取整：kernel 里不必对每像素×每星重复 roundf。
    DeviceBuffer<int> d_star_ix_;
    DeviceBuffer<int> d_star_iy_;
    DeviceBuffer<float> d_star_flux_;
    int star_count_ = 0;
};

}  // namespace telescope
