#include "telescope/fits_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace telescope {
namespace {

constexpr int kCardSize = 80;
constexpr int kBlockSize = 2880;

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// 写一张 FITS 头卡：关键字左对齐 8 列，第 9 列 '='，值区为第 11..30 列。
// 逻辑值/数值按 FITS 标准右对齐（如 SIMPLE 的 T 在第 30 列），
// 字符串（以单引号开头）左对齐。左对齐数值虽然能被 astropy 读，
// 但会触发 "SIMPLE card ... doesn't respect the FITS Standard" 警告。
void write_card(std::FILE* fp, const char* key, const std::string& value) {
    std::array<char, kCardSize> card;
    card.fill(' ');

    const std::size_t klen = std::strlen(key);
    if (klen == 0 || klen > 8) {
        throw std::invalid_argument(std::string("FITS keyword must be 1..8 chars: ") + key);
    }
    // FITS 值区只有 70 列，超长会静默截断成坏卡；这里直接报错。
    if (value.size() > 70) {
        throw std::invalid_argument(std::string("FITS value too long for card: ") + value);
    }
    std::memcpy(card.data(), key, klen);
    card[8] = '=';
    card[9] = ' ';

    const bool quoted = !value.empty() && value.front() == '\'';
    const bool logical = value == "T" || value == "F";
    const bool numeric = !value.empty() &&
                         (std::isdigit(static_cast<unsigned char>(value.front())) ||
                          value.front() == '+' || value.front() == '-' || value.front() == '.');
    if (quoted) {
        std::memcpy(card.data() + 10, value.data(), value.size());
    } else if (logical || numeric) {
        // 值区 20 列：列 11..30，即下标 10..29；右对齐。
        if (value.size() > 20) {
            throw std::invalid_argument(std::string("FITS numeric value too long: ") + value);
        }
        const std::size_t start = 30 - value.size();
        std::memcpy(card.data() + start, value.data(), value.size());
    } else {
        // 其他（如纯字符串但调用方忘了加引号）保持左对齐。
        std::memcpy(card.data() + 10, value.data(), value.size());
    }
    std::fwrite(card.data(), 1, kCardSize, fp);
}

void write_end_card(std::FILE* fp) {
    std::array<char, kCardSize> card;
    card.fill(' ');
    std::memcpy(card.data(), "END", 3);
    std::fwrite(card.data(), 1, kCardSize, fp);
}

void pad_to_block(std::FILE* fp, char fill) {
    const long pos = std::ftell(fp);
    if (pos < 0) return;
    const int rem = static_cast<int>(pos % kBlockSize);
    if (rem == 0) return;
    const int pad = kBlockSize - rem;
    std::vector<char> blanks(static_cast<std::size_t>(pad), fill);
    std::fwrite(blanks.data(), 1, blanks.size(), fp);
}

std::string card_value(const std::string& card) {
    if (card.size() < 9) return "";
    // 标准卡：KEY(8 列) + '=' + value。也容忍 KEY....value 的非标准写法，
    // 但必须从第 9 列开始取，不能先 substr(9) 再找 '='。
    std::string v = card.substr(8);
    const auto eq = v.find('=');
    if (eq != std::string::npos) v = v.substr(eq + 1);
    v = trim(v);

    // FITS 字符串值用单引号包裹；这里去掉引号，方便 strtod/sscanf。
    if (v.size() >= 2 && v.front() == '\'') {
        const auto close = v.rfind('\'');
        if (close != std::string::npos && close > 0) {
            v = trim(v.substr(1, close - 1));
        }
    }
    return v;
}

bool parse_int(const std::string& card, int& out) {
    const std::string v = card_value(card);
    if (v.empty()) return false;
    char* end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str()) return false;
    while (*end == ' ') ++end;
    if (*end != '\0' && *end != '/') return false;  // 允许 FITS 行内注释 "/ ..."
    out = static_cast<int>(x);
    return true;
}

bool parse_float(const std::string& card, float& out) {
    const std::string v = card_value(card);
    if (v.empty()) return false;
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str()) return false;
    while (*end == ' ') ++end;
    if (*end != '\0' && *end != '/') return false;
    out = static_cast<float>(x);
    return true;
}

uint32_t swap_bytes(uint32_t v) {
    return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |
           ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
}

bool host_is_little_endian() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

}  // namespace

bool write_fits_image(const std::string& path,
                      const std::vector<float>& image,
                      const FitsMetadata& meta) {
    if (meta.img_size <= 0) return false;
    const std::size_t expected =
        static_cast<std::size_t>(meta.img_size) * static_cast<std::size_t>(meta.img_size);
    if (image.size() != expected) return false;

    const std::size_t nstars = meta.star_x.size();
    if (meta.star_y.size() != nstars || meta.star_flux.size() != nstars) return false;
    if (nstars > 100) return false;  // STAR00..STAR99

    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    char buf[64];
    write_card(fp, "SIMPLE", "T");
    write_card(fp, "BITPIX", "-32");
    write_card(fp, "NAXIS", "2");
    std::snprintf(buf, sizeof(buf), "%d", meta.img_size);
    write_card(fp, "NAXIS1", buf);
    write_card(fp, "NAXIS2", buf);
    std::snprintf(buf, sizeof(buf), "%.6f", meta.theta_deg);
    write_card(fp, "THETA", buf);
    std::snprintf(buf, sizeof(buf), "%.6f", meta.defocus);
    write_card(fp, "DEFOCUS", buf);
    for (int k = 0; k < kNumZernike; ++k) {
        std::snprintf(buf, sizeof(buf), "%.6f", meta.zernike[k]);
        char key[16];
        std::snprintf(key, sizeof(key), "ZERN%02d", k + kNollFirst);
        write_card(fp, key, buf);
    }
    std::snprintf(buf, sizeof(buf), "%d", meta.seed);
    write_card(fp, "SEED", buf);
    std::snprintf(buf, sizeof(buf), "%.6f", meta.background);
    write_card(fp, "BKG", buf);
    std::snprintf(buf, sizeof(buf), "%.6f", meta.read_noise);
    write_card(fp, "RDNOISE", buf);
    std::snprintf(buf, sizeof(buf), "%zu", nstars);
    write_card(fp, "NSTARS", buf);
    for (std::size_t i = 0; i < nstars; ++i) {
        char key[16];
        std::snprintf(key, sizeof(key), "STAR%02zu", i);
        std::snprintf(buf, sizeof(buf), "'%.6f %.6f %.6f'",
                      meta.star_x[i], meta.star_y[i], meta.star_flux[i]);
        write_card(fp, key, buf);
    }
    write_end_card(fp);
    pad_to_block(fp, ' ');

    const bool little = host_is_little_endian();
    std::vector<float> out(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &image[i], sizeof(bits));
        if (little) bits = swap_bytes(bits);
        std::memcpy(&out[i], &bits, sizeof(bits));
    }
    const std::size_t written = std::fwrite(out.data(), sizeof(float), out.size(), fp);
    pad_to_block(fp, 0);
    const bool io_error = std::ferror(fp) != 0;
    const bool ok = (written == out.size()) && !io_error;
    std::fclose(fp);
    return ok;
}

bool read_fits_image(const std::string& path,
                     std::vector<float>& image,
                     FitsMetadata& meta) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    int naxis = 0;
    int bitpix = 0;
    int naxis1 = 0;
    int naxis2 = 0;
    int nstars_header = -1;
    meta = FitsMetadata{};

    bool found_end = false;
    for (;;) {
        std::array<char, kCardSize + 1> card;
        if (std::fread(card.data(), 1, kCardSize, fp) != kCardSize) break;
        card[kCardSize] = '\0';
        const std::string s(card.data(), kCardSize);

        if (std::strncmp(card.data(), "END", 3) == 0) {
            found_end = true;
            break;
        }
        if (std::strncmp(card.data(), "BITPIX", 6) == 0) parse_int(s, bitpix);
        if (std::strncmp(card.data(), "NAXIS", 5) == 0 &&
            std::strncmp(card.data(), "NAXIS1", 6) != 0 &&
            std::strncmp(card.data(), "NAXIS2", 6) != 0) {
            parse_int(s, naxis);
        }
        if (std::strncmp(card.data(), "NAXIS1", 6) == 0) parse_int(s, naxis1);
        if (std::strncmp(card.data(), "NAXIS2", 6) == 0) parse_int(s, naxis2);
        if (std::strncmp(card.data(), "THETA", 5) == 0) parse_float(s, meta.theta_deg);
        if (std::strncmp(card.data(), "DEFOCUS", 7) == 0) parse_float(s, meta.defocus);
        if (std::strncmp(card.data(), "SEED", 4) == 0) parse_int(s, meta.seed);
        if (std::strncmp(card.data(), "BKG", 3) == 0) parse_float(s, meta.background);
        if (std::strncmp(card.data(), "RDNOISE", 7) == 0) parse_float(s, meta.read_noise);
        if (std::strncmp(card.data(), "NSTARS", 6) == 0) parse_int(s, nstars_header);
        if (std::strncmp(card.data(), "STAR", 4) == 0 &&
            card[4] >= '0' && card[4] <= '9') {
            const int index = std::atoi(card.data() + 4);
            if (index >= 0 && index < 100) {
                float x = 0.0f, y = 0.0f, flux = 0.0f;
                if (std::sscanf(card_value(s).c_str(), "%f %f %f", &x, &y, &flux) == 3) {
                    const std::size_t need = static_cast<std::size_t>(index) + 1;
                    if (meta.star_x.size() < need) {
                        meta.star_x.resize(need);
                        meta.star_y.resize(need);
                        meta.star_flux.resize(need);
                    }
                    meta.star_x[index] = x;
                    meta.star_y[index] = y;
                    meta.star_flux[index] = flux;
                }
            }
        }
        for (int k = 0; k < kNumZernike; ++k) {
            char key[16];
            std::snprintf(key, sizeof(key), "ZERN%02d", k + kNollFirst);
            if (std::strncmp(card.data(), key, 6) == 0) {
                parse_float(s, meta.zernike[k]);
            }
        }
    }
    if (!found_end) {
        std::fclose(fp);
        return false;
    }
    if (nstars_header >= 0 &&
        meta.star_x.size() != static_cast<std::size_t>(nstars_header)) {
        std::fclose(fp);
        return false;
    }
    if (meta.star_y.size() != meta.star_x.size() ||
        meta.star_flux.size() != meta.star_x.size()) {
        std::fclose(fp);
        return false;
    }

    // 跳到 2880 字节块边界。
    const long pos = std::ftell(fp);
    if (pos < 0) {
        std::fclose(fp);
        return false;
    }
    const int rem = static_cast<int>(pos % kBlockSize);
    if (rem != 0 && std::fseek(fp, kBlockSize - rem, SEEK_CUR) != 0) {
        std::fclose(fp);
        return false;
    }

    if (bitpix != -32 || naxis != 2 || naxis1 <= 0 || naxis1 != naxis2) {
        std::fclose(fp);
        return false;
    }

    const int n = naxis1;
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<float> raw(total);
    const std::size_t got = std::fread(raw.data(), sizeof(float), raw.size(), fp);
    std::fclose(fp);
    if (got != total) return false;

    const bool little = host_is_little_endian();
    image.resize(total);
    for (std::size_t i = 0; i < total; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &raw[i], sizeof(bits));
        if (little) bits = swap_bytes(bits);
        std::memcpy(&image[i], &bits, sizeof(bits));
    }
    meta.img_size = n;
    return true;
}

}  // namespace telescope
