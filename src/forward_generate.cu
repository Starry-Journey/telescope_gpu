#include "forward_model.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <sys/stat.h>

// ---------- 简单 FITS 写入（手写头，避免依赖 CCFits） ----------
static void write_fits(const std::string& path,
                       const std::vector<float>& image,
                       int img_size,
                       float theta,
                       float defocus,
                       const float* true_z8)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "无法打开 %s\n", path.c_str());
        return;
    }

    auto write_card = [&](const char* key, const char* value) {
        char buf[80];
        memset(buf, ' ', 80);
        size_t klen = strlen(key);
        if (klen > 8) klen = 8;
        memcpy(buf, key, klen);
        buf[8] = '=';
        buf[9] = ' ';
        size_t vlen = strlen(value);
        if (vlen > 70) vlen = 70;
        memcpy(buf + 10, value, vlen);
        fwrite(buf, 1, 80, fp);
    };

    auto write_end = [&]() {
        char buf[80];
        memset(buf, ' ', 80);
        memcpy(buf, "END", 3);
        fwrite(buf, 1, 80, fp);
    };

    write_card("SIMPLE", "                   T");
    write_card("BITPIX", "                 -32");
    write_card("NAXIS", "                   2");

    char buf[32];
    snprintf(buf, sizeof(buf), "%20d", img_size);
    write_card("NAXIS1", buf);
    snprintf(buf, sizeof(buf), "%20d", img_size);
    write_card("NAXIS2", buf);
    snprintf(buf, sizeof(buf), "%20.6f", theta);
    write_card("THETA", buf);
    snprintf(buf, sizeof(buf), "%20.6f", defocus);
    write_card("DEFOCUS", buf);

    for (int j = 0; j < 8; ++j) {
        char key[8];
        snprintf(key, sizeof(key), "ZERN%02d", j + 2);
        snprintf(buf, sizeof(buf), "%20.6f", true_z8[j]);
        write_card(key, buf);
    }

    write_end();

    // 补齐 2880 边界
    long pos = ftell(fp);
    int pad = 2880 - (int)(pos % 2880);
    if (pad < 2880) {
        std::vector<char> blanks(pad, ' ');
        fwrite(blanks.data(), 1, pad, fp);
    }

    // 写数据（大端序）
    std::vector<float> be_data(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        uint32_t v;
        memcpy(&v, &image[i], 4);
        v = ((v & 0xFF000000) >> 24) |
            ((v & 0x00FF0000) >> 8)  |
            ((v & 0x0000FF00) << 8)  |
            ((v & 0x000000FF) << 24);
        memcpy(&be_data[i], &v, 4);
    }
    fwrite(be_data.data(), 4, be_data.size(), fp);

    // 补齐 2880 倍数
    pos = ftell(fp);
    pad = 2880 - (int)(pos % 2880);
    if (pad < 2880) {
        std::vector<char> blanks(pad, 0);
        fwrite(blanks.data(), 1, pad, fp);
    }

    fclose(fp);
}

// ------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("=== forward_generate ===\n");
    float theta = 45.0f;
    if (argc > 1) theta = atof(argv[1]);
    printf("θ = %.1f°\n", theta);

    OpticalParams params;
    params.img_size = 256;
    params.oversample = 2;
    params.fft_size = params.img_size * params.oversample;
    params.pupil_radius = 0.45f;

    ForwardModelGPU model(params);

    // 该角度的 Zernike 真值
    float z8[8];
    ZernikeTable::get(theta, z8);
    printf("Z4=%.4f Z5=%.4f Z6=%.4f Z7=%.4f Z8=%.4f Z9=%.4f\n",
           z8[2], z8[3], z8[4], z8[5], z8[6], z8[7]);

    // 5 帧对称离焦
    const float PD_DEFOCUS[5] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const int N_FRAMES = 5;

    // 固定随机种子，复刻 Python 版
    srand(42);
    const int STAR_NUM = 15;
    std::vector<float> h_sx(STAR_NUM), h_sy(STAR_NUM), h_sf(STAR_NUM);
    for (int i = 0; i < STAR_NUM; ++i) {
        h_sx[i] = 50.0f + (rand() / (float)RAND_MAX) * (params.img_size - 100);
        h_sy[i] = 50.0f + (rand() / (float)RAND_MAX) * (params.img_size - 100);
        h_sf[i] = 3000.0f + (rand() / (float)RAND_MAX) * 12000.0f;
    }

    // 输出目录
    char dir_name[64];
    snprintf(dir_name, sizeof(dir_name), "theta_%03ddeg", (int)theta);
    mkdir(dir_name, 0755);

    // 生成每帧
    for (int f = 0; f < N_FRAMES; ++f) {
        float defocus = PD_DEFOCUS[f];
        printf("  帧 %d: defocus = %+.1fλ\n", f, defocus);

        std::vector<float> h_image;
        model.generate_image_host(z8, h_sx.data(), h_sy.data(), h_sf.data(),
                                   STAR_NUM, defocus, h_image);

        char fname[256];
        snprintf(fname, sizeof(fname),
                 "%s/star_field_theta%03d_defocus_%+.1fwaves.fits",
                 dir_name, (int)theta, defocus);
        write_fits(fname, h_image, params.img_size, theta, defocus, z8);
        printf("    已保存: %s\n", fname);
    }

    printf("Done.\n");
    return 0;
}