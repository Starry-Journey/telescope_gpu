#include "test_framework.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "telescope/fits_io.hpp"

using namespace telescope;

namespace {

std::string temp_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

FitsMetadata make_metadata() {
    FitsMetadata meta;
    meta.img_size = 8;
    meta.theta_deg = 45.0f;
    meta.defocus = -1.0f;
    meta.zernike = {0.0f, 0.0f, 1.47f, 0.0f, 0.0f, 0.0f, 0.17f, 0.18f};
    meta.seed = 42;
    meta.background = 200.0f;
    meta.read_noise = 5.0f;
    meta.star_x = {10.5f, 20.25f, 30.0f};
    meta.star_y = {11.5f, 21.25f, 31.0f};
    meta.star_flux = {3000.0f, 8000.5f, 15000.0f};
    return meta;
}

}  // namespace

TEST(fits_roundtrip_with_stars) {
    const std::string path = temp_path("telescope_test_roundtrip.fits");
    std::vector<float> image(64);
    for (int i = 0; i < 64; ++i) image[i] = 0.5f * i - 3.0f;

    const FitsMetadata meta = make_metadata();
    REQUIRE(write_fits_image(path, image, meta));

    std::vector<float> loaded;
    FitsMetadata loaded_meta;
    REQUIRE(read_fits_image(path, loaded, loaded_meta));

    CHECK(loaded.size() == image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
        CHECK_NEAR(loaded[i], image[i], 1e-6);
    }

    CHECK(loaded_meta.img_size == meta.img_size);
    CHECK_NEAR(loaded_meta.theta_deg, meta.theta_deg, 1e-5);
    CHECK_NEAR(loaded_meta.defocus, meta.defocus, 1e-5);
    CHECK_NEAR(loaded_meta.background, meta.background, 1e-5);
    CHECK_NEAR(loaded_meta.read_noise, meta.read_noise, 1e-5);
    CHECK(loaded_meta.seed == meta.seed);

    for (int k = 0; k < kNumZernike; ++k) {
        CHECK_NEAR(loaded_meta.zernike[k], meta.zernike[k], 1e-4);
    }

    REQUIRE(loaded_meta.star_x.size() == meta.star_x.size());
    for (std::size_t i = 0; i < meta.star_x.size(); ++i) {
        CHECK_NEAR(loaded_meta.star_x[i], meta.star_x[i], 1e-3);
        CHECK_NEAR(loaded_meta.star_y[i], meta.star_y[i], 1e-3);
        CHECK_NEAR(loaded_meta.star_flux[i], meta.star_flux[i], 1e-2);
    }

    std::filesystem::remove(path);
}

TEST(fits_rejects_wrong_image_size) {
    const std::string path = temp_path("telescope_test_wrong_size.fits");
    std::vector<float> image(10, 1.0f);
    CHECK(!write_fits_image(path, image, make_metadata()));
}

TEST(fits_rejects_mismatched_star_arrays) {
    const std::string path = temp_path("telescope_test_bad_stars.fits");
    std::vector<float> image(64, 1.0f);
    FitsMetadata meta = make_metadata();
    meta.star_y.pop_back();
    CHECK(!write_fits_image(path, image, meta));
}

TEST(fits_read_missing_file) {
    std::vector<float> image;
    FitsMetadata meta;
    CHECK(!read_fits_image(temp_path("telescope_test_does_not_exist.fits"), image, meta));
}
