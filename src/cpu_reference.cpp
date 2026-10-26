#include "telescope/cpu_reference.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

#include "telescope/fft_cpu.hpp"

namespace telescope {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

CpuForwardModel::CpuForwardModel(const OpticalParams& params)
    : params_(params), grid_(make_pupil_grid(params)) {}

void CpuForwardModel::compute_psf(const ZernikeCoeffs& coeffs,
                                  float defocus,
                                  std::vector<float>& psf) const {
    const int n = params_.fft_size();
    const int total = n * n;
    const int img = params_.img_size;
    const int img_total = img * img;

    std::vector<Complex> field(total, Complex(0.0, 0.0));
    for (int idx = 0; idx < total; ++idx) {
        if (!grid_.mask[idx]) continue;
        double phase = 0.0;
        for (int k = 0; k < kNumZernike; ++k) {
            double c = coeffs[k];
            if (k == 2) c += defocus;  // Z4 是离焦项
            phase += c * grid_.basis[static_cast<std::size_t>(k) * total + idx];
        }
        const double angle = 2.0 * kPi * phase;
        field[idx] = Complex(std::cos(angle), std::sin(angle));
    }

    fft2d(field, n, /*inverse=*/false);

    std::vector<double> psf_full(total);
    const double norm = 1.0 / static_cast<double>(total);
    const int half = n / 2;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int sx = (x + half) % n;
            const int sy = (y + half) % n;
            const Complex v = field[sy * n + sx];
            psf_full[y * n + x] = std::norm(v) * norm;
        }
    }

    psf.assign(img_total, 0.0f);
    for (int oy = 0; oy < img; ++oy) {
        for (int ox = 0; ox < img; ++ox) {
            double sum = 0.0;
            for (int dy = 0; dy < params_.oversample; ++dy) {
                for (int dx = 0; dx < params_.oversample; ++dx) {
                    sum += psf_full[(oy * params_.oversample + dy) * n +
                                    (ox * params_.oversample + dx)];
                }
            }
            psf[oy * img + ox] = static_cast<float>(sum);
        }
    }

    double sum = 0.0;
    for (float v : psf) sum += v;
    if (sum > 1e-20) {
        const float inv = static_cast<float>(1.0 / sum);
        for (float& v : psf) v *= inv;
    }
}

void CpuForwardModel::generate_image(const ZernikeCoeffs& coeffs,
                                     const StarField& stars,
                                     float defocus,
                                     std::vector<float>& image) const {
    if (!stars.valid()) throw std::invalid_argument("StarField arrays must have equal size");

    std::vector<float> psf;
    compute_psf(coeffs, defocus, psf);

    const int img = params_.img_size;
    const int img_total = img * img;
    const int half = img / 2;
    image.assign(img_total, 0.0f);

    for (int py = 0; py < img; ++py) {
        for (int px = 0; px < img; ++px) {
            double sum = 0.0;
            for (int s = 0; s < stars.size(); ++s) {
                const int ix = static_cast<int>(std::lround(stars.x[s]));
                const int iy = static_cast<int>(std::lround(stars.y[s]));
                const int psf_x = px - ix + half;
                const int psf_y = py - iy + half;
                if (psf_x >= 0 && psf_x < img && psf_y >= 0 && psf_y < img) {
                    sum += static_cast<double>(stars.flux[s]) *
                           psf[psf_y * img + psf_x];
                }
            }
            image[py * img + px] = static_cast<float>(sum);
        }
    }
}

}  // namespace telescope
