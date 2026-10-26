#include "test_framework.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "telescope/config.hpp"

using namespace telescope;

namespace {

std::string write_temp_config(const std::string& name, const std::string& text) {
    const std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    std::ofstream out(path);
    out << text;
    return path;
}

}  // namespace

TEST(config_load_and_override) {
    const std::string path = write_temp_config(
        "telescope_test_config.conf",
        "# comment\n"
        "img_size = 64\n"
        "oversample = 2\n"
        "pupil_radius = 0.4\n"
        "theta_deg = 30\n"
        "defocus = 0, 1, -1\n"
        "star_count = 8\n"
        "flux_min = 1000\n"
        "flux_max = 2000\n"
        "seed = 7\n"
        "background = 100\n"
        "read_noise = 2.5\n"
        "solve_starts = 4\n"
        "solve_refine = 2\n"
        "solve_seed = 99\n"
        "solve_maxeval = 300\n"
        "data_dir = custom_dir\n"
        "result_file = out.txt\n"
        "report_file = report.txt\n");

    const Config cfg = load_config_file(path);
    CHECK(cfg.optics.img_size == 64);
    CHECK(cfg.optics.oversample == 2);
    CHECK_NEAR(cfg.optics.pupil_radius, 0.4, 1e-6);
    CHECK_NEAR(cfg.theta_deg, 30.0, 1e-6);
    CHECK(cfg.defocus.size() == 3);
    CHECK_NEAR(cfg.defocus[1], 1.0, 1e-6);
    CHECK(cfg.star_count == 8);
    CHECK(cfg.solve_seed == 99);
    CHECK(cfg.data_dir == "custom_dir");
    CHECK(cfg.result_file == "out.txt");

    CliArgs cli;
    cli.config_path = path;
    cli.config_explicit = true;
    cli.overrides["theta_deg"] = "60";
    cli.overrides["img-size"] = "128";
    const Config merged = resolve_config(cli);
    CHECK_NEAR(merged.theta_deg, 60.0, 1e-6);
    CHECK(merged.optics.img_size == 128);
    CHECK(merged.data_dir == "custom_dir");

    std::filesystem::remove(path);
}

TEST(config_rejects_unknown_key) {
    const std::string path = write_temp_config(
        "telescope_test_bad_key.conf", "img_size = 64\ntypo_key = 1\n");
    CHECK_THROWS(load_config_file(path));
    std::filesystem::remove(path);
}

TEST(config_rejects_bad_fft_size) {
    const std::string path = write_temp_config(
        "telescope_test_bad_fft.conf", "img_size = 100\noversample = 2\n");
    CHECK_THROWS(load_config_file(path));
    std::filesystem::remove(path);
}

TEST(config_rejects_bad_defocus) {
    const std::string path = write_temp_config(
        "telescope_test_bad_defocus.conf", "defocus = 0\n");
    CHECK_THROWS(load_config_file(path));
    std::filesystem::remove(path);
}

TEST(config_default_data_dir) {
    Config cfg;
    cfg.theta_deg = 45.0f;
    CHECK(cfg.resolved_data_dir() == "theta_045deg");
    cfg.data_dir = "custom";
    CHECK(cfg.resolved_data_dir() == "custom");
}

TEST(cli_parsing) {
    char prog[] = "app";
    char arg1[] = "--config";
    char arg2[] = "my.conf";
    char arg3[] = "--theta";
    char arg4[] = "55";
    char arg5[] = "--star-count=9";
    char* argv[] = {prog, arg1, arg2, arg3, arg4, arg5};
    const CliArgs cli = parse_cli(6, argv);
    CHECK(cli.config_explicit);
    CHECK(cli.config_path == "my.conf");
    CHECK(cli.overrides.at("theta") == "55");
    CHECK(cli.overrides.at("star-count") == "9");

    char prog2[] = "app";
    char pos[] = "45";
    char* argv2[] = {prog2, pos};
    const CliArgs cli2 = parse_cli(2, argv2);
    CHECK(cli2.overrides.at("theta_deg") == "45");
}

TEST(cli_rejects_missing_value) {
    char prog[] = "app";
    char arg[] = "--theta";
    char* argv[] = {prog, arg};
    CHECK_THROWS(parse_cli(2, argv));
}
