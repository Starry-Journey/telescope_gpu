#pragma once

/// @file data.hpp
/// @brief Common data contracts for the instrument inversion framework.
///
/// The framework is deliberately independent of FITS/CUDA/Zemax.  A frontend
/// adapter converts telescope/CCD products into these types; a backend adapter
/// (phase diversity, synthetic model, ...) consumes them.

#include <map>
#include <string>
#include <vector>

namespace instrument {

/// A 2-D detector image in a declared unit ("e-", "ADU", "wave", ...).
struct Image {
    int width = 0;
    int height = 0;
    std::string unit;
    std::vector<float> pixels;  // row-major, size = width*height

    bool valid() const {
        return width > 0 && height > 0 &&
               pixels.size() == static_cast<std::size_t>(width) * height;
    }
};

struct StarPoint {
    float x = 0.0f;
    float y = 0.0f;
    float flux = 0.0f;
};

/// One exposure/frame with the metadata needed by reduction and inversion.
struct Frame {
    Image image;
    double exposure_s = 1.0;
    std::map<std::string, double> scalars;  // defocus, theta, background, ...
    std::string quality;                    // "ok", "saturated", ...
};

/// A coherent observation: multiple frames of the same field/parameter set.
struct Observation {
    std::string id;
    std::vector<Frame> frames;
    std::vector<double> defocus;       // waves, one per frame
    std::vector<StarPoint> stars;      // shared star table
    std::vector<double> truth;         // optional ground truth, empty if unknown
    std::map<std::string, double> scalars;
};

/// Lightweight status for public API boundaries.  Exceptions are reserved for
/// programmer errors inside an implementation; data/IO failures use Status.
struct Status {
    bool ok = true;
    std::string message;

    static Status success() { return {}; }
    static Status failure(std::string why) { return {false, std::move(why)}; }
    explicit operator bool() const { return ok; }
};

}  // namespace instrument
