#pragma once

/// @file fits_reader.hpp
/// @brief Adapter: FITS frames (the existing telescope_gpu format) -> Observation.
///
/// The reader does not implement FITS; it reuses telescope::read_fits_image.
/// This is the upstream boundary: CCD/pipeline or forward_generate can write
/// FITS, and the framework sees the same Observation contract as synthetic data.

#include <string>
#include <vector>

#include "instrument/data.hpp"

namespace instrument {

Status read_fits_observation(const std::vector<std::string>& paths,
                             Observation& observation);

}  // namespace instrument
