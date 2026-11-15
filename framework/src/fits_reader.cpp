#include "instrument/fits_reader.hpp"

#include <stdexcept>

#include "telescope/fits_io.hpp"

namespace instrument {

Status read_fits_observation(const std::vector<std::string>& paths,
                             Observation& observation) {
    if (paths.empty()) return Status::failure("no FITS paths");
    observation = Observation{};
    observation.id = "fits";

    bool stars_loaded = false;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        std::vector<float> pixels;
        telescope::FitsMetadata meta;
        try {
            if (!telescope::read_fits_image(paths[i], pixels, meta)) {
                return Status::failure("cannot read FITS: " + paths[i]);
            }
        } catch (const std::exception& e) {
            return Status::failure("FITS error: " + paths[i] + ": " + e.what());
        }
        if (i == 0) {
            if (!meta.star_x.empty()) {
                observation.stars.reserve(meta.star_x.size());
                for (std::size_t s = 0; s < meta.star_x.size(); ++s) {
                    StarPoint p;
                    p.x = meta.star_x[s];
                    p.y = meta.star_y[s];
                    p.flux = meta.star_flux[s];
                    observation.stars.push_back(p);
                }
                stars_loaded = true;
            }
            observation.truth.assign(meta.zernike.begin(), meta.zernike.end());
        }

        Frame f;
        f.image.width = meta.img_size;
        f.image.height = meta.img_size;
        f.image.unit = "e-";
        f.image.pixels = std::move(pixels);
        if (!f.image.valid()) {
            return Status::failure("invalid FITS image shape: " + paths[i]);
        }
        f.exposure_s = 1.0;
        f.scalars["defocus"] = static_cast<double>(meta.defocus);
        f.scalars["theta"] = static_cast<double>(meta.theta_deg);
        f.scalars["background"] = static_cast<double>(meta.background);
        f.scalars["read_noise"] = static_cast<double>(meta.read_noise);
        f.quality = "ok";
        observation.frames.push_back(std::move(f));
        observation.defocus.push_back(static_cast<double>(meta.defocus));
    }
    if (!stars_loaded) {
        return Status::failure("FITS header has no STARxx table");
    }
    observation.scalars["n_frames"] = static_cast<double>(observation.frames.size());
    return Status::success();
}

}  // namespace instrument
