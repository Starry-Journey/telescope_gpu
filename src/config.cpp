#include "telescope/config.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace telescope {
namespace {

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string normalize_key(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    return s;
}

float to_float(const std::string& value, const std::string& where) {
    try {
        std::size_t used = 0;
        const float v = std::stof(value, &used);
        if (used != value.size()) throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(where + ": expected float, got '" + value + "'");
    }
}

int to_int(const std::string& value, const std::string& where) {
    try {
        std::size_t used = 0;
        const int v = std::stoi(value, &used);
        if (used != value.size()) throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(where + ": expected int, got '" + value + "'");
    }
}

unsigned to_uint(const std::string& value, const std::string& where) {
    // std::stoul("-1") 会回绕成 ULONG_MAX 而不抛异常，必须显式拒绝负号。
    if (!value.empty() && value[0] == '-') {
        throw std::runtime_error(where + ": expected unsigned int, got '" + value + "'");
    }
    try {
        std::size_t used = 0;
        const unsigned long v = std::stoul(value, &used);
        if (used != value.size()) throw std::invalid_argument("trailing characters");
        return static_cast<unsigned>(v);
    } catch (const std::exception&) {
        throw std::runtime_error(where + ": expected unsigned int, got '" + value + "'");
    }
}

std::vector<float> to_float_list(const std::string& value, const std::string& where) {
    std::vector<float> out;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t end = value.find(',', begin);
        const std::string item = trim(value.substr(begin, end == std::string::npos
                                                               ? std::string::npos
                                                               : end - begin));
        if (item.empty()) throw std::runtime_error(where + ": empty list item");
        out.push_back(to_float(item, where));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return out;
}

void set_value(Config& cfg, const std::string& raw_key, const std::string& raw_value,
               const std::string& where) {
    const std::string key = normalize_key(raw_key);
    const std::string value = trim(raw_value);

    if (key == "img_size") {
        cfg.optics.img_size = to_int(value, where);
    } else if (key == "oversample") {
        cfg.optics.oversample = to_int(value, where);
    } else if (key == "pupil_radius") {
        cfg.optics.pupil_radius = to_float(value, where);
    } else if (key == "theta" || key == "theta_deg") {
        cfg.theta_deg = to_float(value, where);
    } else if (key == "defocus") {
        cfg.defocus = to_float_list(value, where);
    } else if (key == "star_count") {
        cfg.star_count = to_int(value, where);
    } else if (key == "flux_min") {
        cfg.flux_min = to_float(value, where);
    } else if (key == "flux_max") {
        cfg.flux_max = to_float(value, where);
    } else if (key == "seed") {
        cfg.seed = to_int(value, where);
    } else if (key == "background") {
        cfg.background = to_float(value, where);
    } else if (key == "read_noise") {
        cfg.read_noise = to_float(value, where);
    } else if (key == "background_mode") {
        cfg.background_mode = normalize_key(value);
    } else if (key == "solve_starts") {
        cfg.solve_starts = to_int(value, where);
    } else if (key == "solve_refine") {
        cfg.solve_refine = to_int(value, where);
    } else if (key == "solve_seed") {
        cfg.solve_seed = to_uint(value, where);
    } else if (key == "solve_maxeval") {
        cfg.solve_maxeval = to_int(value, where);
    } else if (key == "data_dir") {
        cfg.data_dir = value;
    } else if (key == "result_file") {
        cfg.result_file = value;
    } else if (key == "report_file") {
        cfg.report_file = value;
    } else {
        throw std::runtime_error(where + ": unknown key '" + raw_key + "'");
    }
}

void validate(const Config& cfg) {
    if (cfg.optics.img_size <= 0) throw std::runtime_error("img_size must be positive");
    if (cfg.optics.oversample <= 0) throw std::runtime_error("oversample must be positive");
    if (cfg.optics.img_size > 4096 || cfg.optics.oversample > 8) {
        throw std::runtime_error("img_size/oversample too large (img_size <= 4096, oversample <= 8)");
    }
    if (!(cfg.optics.pupil_radius > 0.0f && cfg.optics.pupil_radius <= 1.0f)) {
        throw std::runtime_error("pupil_radius must be in (0, 1]");
    }
    const int n = cfg.optics.fft_size();
    if (n <= 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("img_size * oversample must be a power of two "
                                 "(CPU reference FFT is radix-2)");
    }
    if (!std::isfinite(cfg.theta_deg)) throw std::runtime_error("theta must be finite");
    if (cfg.defocus.size() < 2 || cfg.defocus.size() > static_cast<std::size_t>(kMaxFrames)) {
        throw std::runtime_error("defocus needs 2..16 frames");
    }
    for (float d : cfg.defocus) {
        if (!std::isfinite(d)) throw std::runtime_error("defocus values must be finite");
    }
    // FITS 头部卡片是 STAR00..STAR99，超过 100 颗星写不进去。
    if (cfg.star_count < 0 || cfg.star_count > 100) {
        throw std::runtime_error("star_count must be in 0..100 (FITS STARnn cards)");
    }
    if (!std::isfinite(cfg.flux_min) || !std::isfinite(cfg.flux_max)) {
        throw std::runtime_error("flux_min/flux_max must be finite");
    }
    if (cfg.flux_min < 0.0f) throw std::runtime_error("flux_min must be >= 0");
    if (cfg.flux_min > cfg.flux_max) throw std::runtime_error("flux_min must be <= flux_max");
    // poisson_distribution 的均值是 double，但转成 int 计数前要有物理上界。
    if (cfg.flux_max + cfg.background > 1.0e8f) {
        throw std::runtime_error("flux_max + background too large for Poisson sampling");
    }
    if (!std::isfinite(cfg.background) || cfg.background < 0.0f) {
        throw std::runtime_error("background must be finite and >= 0");
    }
    if (!std::isfinite(cfg.read_noise) || cfg.read_noise < 0.0f) {
        throw std::runtime_error("read_noise must be finite and >= 0");
    }
    if (cfg.background_mode != "header" && cfg.background_mode != "median") {
        throw std::runtime_error("background_mode must be 'header' or 'median'");
    }
    if (cfg.solve_starts < 1) throw std::runtime_error("solve_starts must be >= 1");
    if (cfg.solve_refine < 1) throw std::runtime_error("solve_refine must be >= 1");
    if (cfg.solve_maxeval < 1) throw std::runtime_error("solve_maxeval must be >= 1");
}

}  // namespace

std::string Config::resolved_data_dir() const {
    if (!data_dir.empty()) return data_dir;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "theta_%03ddeg",
                  static_cast<int>(std::lround(theta_deg)));
    return std::string(buf);
}

Config load_config_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    Config cfg;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": expected key = value");
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": empty key");
        }
        // 允许空 value（例如 data_dir = 表示自动命名）；数值键会由
        // set_value 报出清晰的解析错误。
        set_value(cfg, key, value, path + ":" + std::to_string(line_no));
    }
    validate(cfg);
    return cfg;
}

CliArgs parse_cli(int argc, char** argv) {
    CliArgs cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            cli.help = true;
            continue;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) throw std::runtime_error("--config needs a path");
            cli.config_path = argv[++i];
            cli.config_explicit = true;
            continue;
        }
        if (arg.rfind("--config=", 0) == 0) {
            cli.config_path = arg.substr(9);
            cli.config_explicit = true;
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            const std::string body = arg.substr(2);
            const auto eq = body.find('=');
            if (eq != std::string::npos) {
                cli.overrides[body.substr(0, eq)] = body.substr(eq + 1);
            } else {
                if (i + 1 >= argc) {
                    throw std::runtime_error("option --" + body + " needs a value");
                }
                cli.overrides[body] = argv[++i];
            }
            continue;
        }
        if (!arg.empty() && arg[0] != '-') {
            // 兼容旧用法：./forward_generate 45
            cli.overrides["theta_deg"] = arg;
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }
    return cli;
}

Config resolve_config(const CliArgs& cli) {
    Config cfg;
    const bool exists = std::filesystem::exists(cli.config_path);
    if (exists) {
        cfg = load_config_file(cli.config_path);
    } else if (cli.config_explicit) {
        throw std::runtime_error("config file not found: " + cli.config_path);
    }

    for (const auto& [key, value] : cli.overrides) {
        set_value(cfg, key, value, "command line");
    }
    validate(cfg);
    return cfg;
}

void print_usage(const char* program) {
    std::printf(
        "Usage: %s [options] [theta_deg]\n"
        "\n"
        "Options:\n"
        "  --config PATH       read key=value config (default: configs/default.conf)\n"
        "  --theta DEG         observation angle in degrees\n"
        "  --data-dir PATH     directory for FITS input/output\n"
        "  --key value         any key from configs/default.conf\n"
        "  -h, --help          show this help\n"
        "\n"
        "Examples:\n"
        "  %s --theta 45\n"
        "  %s 45 --data-dir theta_045deg\n",
        program, program, program);
}

}  // namespace telescope
