#include "telescope/forward_model.cuh"

#include <algorithm>
#include <stdexcept>

#include "telescope/zernike.hpp"

namespace telescope {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kBlock = 256;

__global__ void pupil_frames_kernel(cufftComplex* pupil,
                                    const float* basis,
                                    const float* coeffs,
                                    const unsigned char* mask,
                                    const float* defocus,
                                    int fft_size,
                                    int n_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fft_size * fft_size;
    const int frame = blockIdx.y;
    if (idx >= total || frame >= n_frames) return;

    const int pos = frame * total + idx;
    if (!mask[idx]) {
        pupil[pos] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    float phase = 0.0f;
    for (int k = 0; k < kNumZernike; ++k) {
        float c = coeffs[k];
        if (k == 2) c += defocus[frame];  // Z4 = defocus
        phase += c * basis[k * total + idx];
    }

    const float angle = 2.0f * kPi * phase;
    pupil[pos] = make_cuFloatComplex(cosf(angle), sinf(angle));
}

// PSF = |FFT|^2 / N^2，含 fftshift（把零频移到图像中心）。
__global__ void psf_frames_kernel(float* psf,
                                  const cufftComplex* fft_out,
                                  int fft_size,
                                  int n_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fft_size * fft_size;
    const int frame = blockIdx.y;
    if (idx >= total || frame >= n_frames) return;

    const int y = idx / fft_size;
    const int x = idx % fft_size;
    const int half = fft_size / 2;
    const int sy = (y + half) % fft_size;
    const int sx = (x + half) % fft_size;
    const int src = frame * total + sy * fft_size + sx;

    const float re = fft_out[src].x;
    const float im = fft_out[src].y;
    psf[frame * total + idx] = (re * re + im * im) / static_cast<float>(total);
}

__global__ void downsample_frames_kernel(float* psf_down,
                                         const float* psf_full,
                                         int img_size,
                                         int oversample,
                                         int n_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int img_total = img_size * img_size;
    const int frame = blockIdx.y;
    if (idx >= img_total || frame >= n_frames) return;

    const int fft_size = img_size * oversample;
    const int oy = idx / img_size;
    const int ox = idx % img_size;

    float sum = 0.0f;
    const float* src = psf_full + frame * fft_size * fft_size;
    for (int dy = 0; dy < oversample; ++dy) {
        for (int dx = 0; dx < oversample; ++dx) {
            sum += src[(oy * oversample + dy) * fft_size + ox * oversample + dx];
        }
    }
    psf_down[frame * img_total + idx] = sum;
}

__global__ void overlay_frames_kernel(float* images,
                                      const float* psf_frames,
                                      const float* star_x,
                                      const float* star_y,
                                      const float* star_flux,
                                      int n_stars,
                                      int n_frames,
                                      int img_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int img_total = img_size * img_size;
    const int frame = blockIdx.y;
    if (idx >= img_total || frame >= n_frames) return;

    const int py = idx / img_size;
    const int px = idx % img_size;
    const int half = img_size / 2;
    const float* psf = psf_frames + frame * img_total;

    float sum = 0.0f;
    for (int s = 0; s < n_stars; ++s) {
        const int ix = static_cast<int>(roundf(star_x[s]));
        const int iy = static_cast<int>(roundf(star_y[s]));
        const int psf_x = px - ix + half;
        const int psf_y = py - iy + half;
        if (psf_x >= 0 && psf_x < img_size && psf_y >= 0 && psf_y < img_size) {
            sum += star_flux[s] * psf[psf_y * img_size + psf_x];
        }
    }
    images[frame * img_total + idx] = sum;
}

// 每个 block 负责一帧，block 内归约求和。
__global__ void reduce_sum_kernel(const float* data,
                                  float* sums,
                                  int total,
                                  int n_frames) {
    const int frame = blockIdx.x;
    if (frame >= n_frames) return;

    const float* p = data + static_cast<std::size_t>(frame) * total;
    float s = 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) s += p[i];

    __shared__ float smem[kBlock];
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[frame] = smem[0];
}

__global__ void scale_frames_kernel(float* data,
                                    const float* sums,
                                    int total,
                                    int n_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_frames * total;
    if (idx >= n) return;

    const int frame = idx / total;
    const float s = sums[frame];
    data[idx] *= (s > 1e-20f) ? (1.0f / s) : 1.0f;
}

}  // namespace

ForwardModelGPU::ForwardModelGPU(const OpticalParams& params, int max_frames)
    : params_(params), max_frames_(max_frames) {
    if (max_frames_ < 1) throw std::invalid_argument("max_frames must be >= 1");

    const int n = params_.fft_size();
    const int total = params_.fft_total();
    const int img_total = params_.img_total();
    const std::size_t frames = static_cast<std::size_t>(max_frames_);

    // 光瞳网格与 Zernike 基函数只在 CPU 端计算一次。
    const PupilGrid grid = make_pupil_grid(params_);

    d_mask_.allocate(static_cast<std::size_t>(total));
    d_basis_.allocate(static_cast<std::size_t>(kNumZernike) * total);
    TEL_CUDA_CHECK(cudaMemcpy(d_mask_.get(), grid.mask.data(),
                              static_cast<std::size_t>(total) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    TEL_CUDA_CHECK(cudaMemcpy(d_basis_.get(), grid.basis.data(),
                              static_cast<std::size_t>(kNumZernike) * total * sizeof(float),
                              cudaMemcpyHostToDevice));

    d_pupil_.allocate(frames * static_cast<std::size_t>(total));
    d_fft_.allocate(frames * static_cast<std::size_t>(total));
    d_psf_full_.allocate(frames * static_cast<std::size_t>(total));
    d_psf_down_.allocate(frames * static_cast<std::size_t>(img_total));
    d_coeffs_.allocate(kNumZernike);
    d_defocus_.allocate(frames);
    d_sums_.allocate(frames);

    TEL_CUFFT_CHECK(cufftPlan2d(&plan_single_, n, n, CUFFT_C2C));

    int dims[2] = {n, n};
    TEL_CUFFT_CHECK(cufftPlanMany(&plan_batch_, 2, dims,
                                  dims, 1, total,
                                  dims, 1, total,
                                  CUFFT_C2C, max_frames_));
}

ForwardModelGPU::~ForwardModelGPU() {
    if (plan_batch_) cufftDestroy(plan_batch_);
    if (plan_single_) cufftDestroy(plan_single_);
}

void ForwardModelGPU::set_stars(const StarField& stars) {
    if (!stars.valid()) {
        throw std::invalid_argument("StarField arrays must have equal size");
    }
    star_count_ = stars.size();
    if (star_count_ == 0) {
        d_star_x_.reset();
        d_star_y_.reset();
        d_star_flux_.reset();
        return;
    }

    const std::size_t bytes = static_cast<std::size_t>(star_count_) * sizeof(float);
    d_star_x_.allocate(star_count_);
    d_star_y_.allocate(star_count_);
    d_star_flux_.allocate(star_count_);
    TEL_CUDA_CHECK(cudaMemcpy(d_star_x_.get(), stars.x.data(), bytes, cudaMemcpyHostToDevice));
    TEL_CUDA_CHECK(cudaMemcpy(d_star_y_.get(), stars.y.data(), bytes, cudaMemcpyHostToDevice));
    TEL_CUDA_CHECK(cudaMemcpy(d_star_flux_.get(), stars.flux.data(), bytes, cudaMemcpyHostToDevice));
}

void ForwardModelGPU::run_psf_frames(const ZernikeCoeffs& coeffs,
                                     const float* h_defocus,
                                     int n_frames) {
    if (n_frames < 1 || n_frames > max_frames_) {
        throw std::invalid_argument("n_frames out of range");
    }

    const int total = params_.fft_total();
    const int img_total = params_.img_total();

    TEL_CUDA_CHECK(cudaMemcpy(d_coeffs_.get(), coeffs.data(),
                              kNumZernike * sizeof(float), cudaMemcpyHostToDevice));
    TEL_CUDA_CHECK(cudaMemcpy(d_defocus_.get(), h_defocus,
                              static_cast<std::size_t>(n_frames) * sizeof(float),
                              cudaMemcpyHostToDevice));

    const dim3 fft_grid(div_up(total, kBlock), n_frames);
    pupil_frames_kernel<<<fft_grid, kBlock>>>(
        d_pupil_.get(), d_basis_.get(), d_coeffs_.get(), d_mask_.get(),
        d_defocus_.get(), params_.fft_size(), n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());

    if (n_frames == 1) {
        TEL_CUFFT_CHECK(cufftExecC2C(plan_single_, d_pupil_.get(), d_fft_.get(),
                                     CUFFT_FORWARD));
    } else {
        // planMany 在构造时固定 batch = max_frames_；多算的帧会被后续 kernel 忽略。
        TEL_CUFFT_CHECK(cufftExecC2C(plan_batch_, d_pupil_.get(), d_fft_.get(),
                                     CUFFT_FORWARD));
    }

    psf_frames_kernel<<<fft_grid, kBlock>>>(
        d_psf_full_.get(), d_fft_.get(), params_.fft_size(), n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());

    const dim3 img_grid(div_up(img_total, kBlock), n_frames);
    downsample_frames_kernel<<<img_grid, kBlock>>>(
        d_psf_down_.get(), d_psf_full_.get(), params_.img_size,
        params_.oversample, n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());

    reduce_sum_kernel<<<n_frames, kBlock>>>(
        d_psf_down_.get(), d_sums_.get(), img_total, n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());
    scale_frames_kernel<<<div_up(n_frames * img_total, kBlock), kBlock>>>(
        d_psf_down_.get(), d_sums_.get(), img_total, n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());
}

void ForwardModelGPU::compute_psf(const ZernikeCoeffs& coeffs,
                                  float defocus,
                                  float* d_psf_out) {
    if (!d_psf_out) throw std::invalid_argument("d_psf_out is null");
    run_psf_frames(coeffs, &defocus, 1);
    TEL_CUDA_CHECK(cudaMemcpy(d_psf_out, d_psf_down_.get(),
                              static_cast<std::size_t>(params_.img_total()) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
}

void ForwardModelGPU::generate_image(const ZernikeCoeffs& coeffs,
                                     float defocus,
                                     float* d_image_out) {
    if (!d_image_out) throw std::invalid_argument("d_image_out is null");
    run_psf_frames(coeffs, &defocus, 1);

    const int img_total = params_.img_total();
    const dim3 grid(div_up(img_total, kBlock), 1);
    overlay_frames_kernel<<<grid, kBlock>>>(
        d_image_out, d_psf_down_.get(),
        d_star_x_.get(), d_star_y_.get(), d_star_flux_.get(),
        star_count_, 1, params_.img_size);
    TEL_CUDA_CHECK(cudaGetLastError());
}

void ForwardModelGPU::generate_frames(const ZernikeCoeffs& coeffs,
                                      const float* h_defocus,
                                      int n_frames,
                                      float* d_frames_out) {
    if (!d_frames_out) throw std::invalid_argument("d_frames_out is null");
    if (!h_defocus) throw std::invalid_argument("h_defocus is null");

    run_psf_frames(coeffs, h_defocus, n_frames);

    const int img_total = params_.img_total();
    const dim3 grid(div_up(img_total, kBlock), n_frames);
    overlay_frames_kernel<<<grid, kBlock>>>(
        d_frames_out, d_psf_down_.get(),
        d_star_x_.get(), d_star_y_.get(), d_star_flux_.get(),
        star_count_, n_frames, params_.img_size);
    TEL_CUDA_CHECK(cudaGetLastError());
}

void ForwardModelGPU::normalize_frames(float* d_frames, int n_frames) {
    if (!d_frames) throw std::invalid_argument("d_frames is null");
    if (n_frames < 1 || n_frames > max_frames_) {
        throw std::invalid_argument("n_frames out of range");
    }

    const int img_total = params_.img_total();
    reduce_sum_kernel<<<n_frames, kBlock>>>(d_frames, d_sums_.get(), img_total, n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());
    scale_frames_kernel<<<div_up(n_frames * img_total, kBlock), kBlock>>>(
        d_frames, d_sums_.get(), img_total, n_frames);
    TEL_CUDA_CHECK(cudaGetLastError());
}

}  // namespace telescope
