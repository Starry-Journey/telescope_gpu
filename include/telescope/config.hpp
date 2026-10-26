#pragma once

#include <map>
#include <string>
#include <vector>

#include "telescope/types.hpp"

namespace telescope {

// 运行参数。所有可在配置文件中设置的值都集中在这里，
// app 只负责“读配置 -> 调库 -> 写结果”，不再散落魔法数字。
struct Config {
    OpticalParams optics{};

    float theta_deg = 45.0f;
    std::vector<float> defocus = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};

    int star_count = 15;
    float flux_min = 3000.0f;
    float flux_max = 15000.0f;
    int seed = 42;
    float background = 200.0f;
    float read_noise = 5.0f;
    // "header": 用 FITS 头部 BKG；"median": 用每帧中位数。
    std::string background_mode = "header";

    int solve_starts = 40;
    int solve_refine = 3;
    unsigned solve_seed = 12345;
    int solve_maxeval = 2000;

    // 空字符串表示使用 theta_%03ddeg。
    std::string data_dir;
    std::string result_file = "inverse_result.txt";
    std::string report_file = "inverse_report.txt";

    std::string resolved_data_dir() const;
};

// 命令行：--config FILE, --key value, --key=value, 以及位置参数 theta。
struct CliArgs {
    std::string config_path = "configs/default.conf";
    bool config_explicit = false;
    bool help = false;
    std::map<std::string, std::string> overrides;
};

CliArgs parse_cli(int argc, char** argv);

// 读取 key = value 配置文件；失败抛 std::runtime_error。
Config load_config_file(const std::string& path);

// 合并优先级：默认值 < 配置文件 < 命令行覆盖。
// 若 cli.config_path 不存在且不是显式指定，则静默使用默认值（方便在 build/ 下跑）。
Config resolve_config(const CliArgs& cli);

void print_usage(const char* program);

}  // namespace telescope
