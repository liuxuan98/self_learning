#include <iostream>
#include "immintrin.h"
#include <vector>

#define B_RATION 29  // 0.114 * 256
#define G_RATION 150 // 0.587 * 256
#define R_RATION 77  // 0.299 * 256

static __m128i v_shuffle_bgr_b0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
static __m128i v_shuffle_bgr_g0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
static __m128i v_shuffle_bgr_r0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
static __m128i v_shuffle_bgr_b1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
static __m128i v_shuffle_bgr_g1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
static __m128i v_shuffle_bgr_r1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
static __m128i v_shuffle_bgr_b2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static __m128i v_shuffle_bgr_g2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static __m128i v_shuffle_bgr_r2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
void print_m256i(__m256i vec)
{
    int16_t arr[16]; // 最好写int32_t，否则会越界
    _mm256_storeu_si256((__m256i *)arr, vec);

    for (int i = 0; i < 16; ++i)
    {
        std::cout << static_cast<int>(arr[i]) << " ";
        if ((i + 1) % 16 == 0)
            std::cout << "| "; // 分隔每个32位边界，便于阅读
    }
    std::cout << std::endl;
}

void print_m256i_u64(__m256i vec)
{
    uint64_t arr[4];
    _mm256_storeu_si256((__m256i *)arr, vec);
    for (int i = 0; i < 4; ++i)
    {
        std::cout << arr[i] << " ";
        if ((i + 1) % 4 == 0)
            std::cout << "| ";
    }
    std::cout << std::endl;
}
void print_m256i_u8(__m256i vec)
{
    uint8_t arr[32];
    _mm256_storeu_si256((__m256i *)arr, vec);
    for (int i = 0; i < 32; ++i)
    {
        std::cout << static_cast<int>(arr[i]) << " ";
        if ((i + 1) % 16 == 0)
            std::cout << "| "; // 分隔每个32位边界，便于阅读
    }
    std::cout << std::endl;
}
void print_m128i(__m128i vec)
{
    uint8_t buffer[16];

    _mm_storeu_si128((__m128i *)buffer, vec);

    std::cout << "buffer values: ";
    for (int i = 0; i < 16; ++i)
    {
        std::cout << static_cast<int>(buffer[i]) << " ";
    }
    std::cout << "\n";
}
inline void vld3_u8x16_avx(const uint8_t *src, __m128i *b, __m128i *g, __m128i *r)
{
    const __m128i bgr0 = _mm_loadu_si128((const __m128i *)src);
    const __m128i bgr1 = _mm_loadu_si128((const __m128i *)(src + 16));
    const __m128i bgr2 = _mm_loadu_si128((const __m128i *)(src + 32));

    print_m128i(bgr0);
    print_m128i(bgr1);
    print_m128i(bgr2);

    const __m128i b0 = _mm_shuffle_epi8(bgr0, v_shuffle_bgr_b0);
    print_m128i(b0);
    const __m128i g0 = _mm_shuffle_epi8(bgr0, v_shuffle_bgr_g0);
    print_m128i(g0);
    const __m128i r0 = _mm_shuffle_epi8(bgr0, v_shuffle_bgr_r0);
    print_m128i(r0);

    const __m128i b1 = _mm_shuffle_epi8(bgr1, v_shuffle_bgr_b1);
    print_m128i(b1);
    const __m128i g1 = _mm_shuffle_epi8(bgr1, v_shuffle_bgr_g1);
    print_m128i(g1);
    const __m128i r1 = _mm_shuffle_epi8(bgr1, v_shuffle_bgr_r1);
    print_m128i(r1);

    const __m128i b2 = _mm_shuffle_epi8(bgr2, v_shuffle_bgr_b2);
    print_m128i(b2);
    const __m128i g2 = _mm_shuffle_epi8(bgr2, v_shuffle_bgr_g2);
    print_m128i(g2);
    const __m128i r2 = _mm_shuffle_epi8(bgr2, v_shuffle_bgr_r2);
    print_m128i(r2);

    *b = _mm_or_si128(_mm_or_si128(b0, b1), b2); // 是SSE指令集中用于对两个128位整数向量的对应字节执行按位逻辑或（OR）操作的函数
    *g = _mm_or_si128(_mm_or_si128(g0, g1), g2);
    *r = _mm_or_si128(_mm_or_si128(r0, r1), r2);
}
/// @brief cl /EHsc /O2 /arch:AVX2 avx_RGB2Gary.cc
/// @return
int main()
{

    std::vector<uint8_t> rgb_data(48); // 48个数 48/3 =16.
    for (size_t i = 0; i < rgb_data.size(); i += 3)
    {
        rgb_data[i] = 1;
        rgb_data[i + 1] = 2;
        rgb_data[i + 2] = 3;
    }

    std::vector<uint8_t> gray_data(16);
    bool rgb_mode_ = true;
    // gray = 0.299*R + 0.587*G + 0.114*B
    __m256i v_b_factor = _mm256_set1_epi16(B_RATION); // 0.114 * 256 = 29.184 约等于 29
    __m256i v_g_factor = _mm256_set1_epi16(G_RATION); // 0.587 * 256 = 150.272 约等于 150
    __m256i v_r_factor = _mm256_set1_epi16(R_RATION); // 0.299 * 256.
    //_mm256_set1_epi16 创建一个256位向量,每个元素都是16位整数,值为value.
    print_m256i(v_b_factor);
    print_m256i(v_g_factor);
    print_m256i(v_r_factor);
    int b_factor = B_RATION, g_factor = G_RATION, r_factor = R_RATION;
    if (rgb_mode_)
    {
        // rgb 格式
        v_b_factor = _mm256_set1_epi16(R_RATION);
        v_r_factor = _mm256_set1_epi16(B_RATION);
        b_factor = R_RATION, r_factor = B_RATION;
    }

    const __m256i v_half = _mm256_set1_epi16(128);
    print_m256i(v_half);
    __m256i v_zero = _mm256_setzero_si256(); // save middle value vector. 8个32位整数或16个16位整数.32个8位整数.
    print_m256i(v_zero);
    const int split_mask = 0b01111000; // 0b01111000 = 0x78 = 0x78

    const uint8_t *src = rgb_data.data();
    uint8_t *dst = gray_data.data();
    // 在宽度范围内的移动地址操作.
    __m128i v_b_u8, v_g_u8, v_r_u8; // 128位向量,8个16位int(),16个8位(uint8_t unsigned char)int.
    // Load RGB to 128 bit vector.
    vld3_u8x16_avx(src, &v_b_u8, &v_g_u8, &v_r_u8); // 将rgb分别加载到3个128位向量中.8*16
    print_m128i(v_b_u8);
    print_m128i(v_g_u8);
    print_m128i(v_r_u8);
    // 16*16
    __m256i v_b_u16 = _mm256_cvtepu8_epi16(v_b_u8); // 16个无符号8位整数(unsigned char or uint8_t)向量转换为16位(short or int16_t)整数向量.
    print_m256i(v_b_u16);
    __m256i v_g_u16 = _mm256_cvtepu8_epi16(v_g_u8); // AVX2指令集
    print_m256i(v_g_u16);
    __m256i v_r_u16 = _mm256_cvtepu8_epi16(v_r_u8);
    print_m256i(v_r_u16);

    __m256i v_b_product = _mm256_mullo_epi16(v_b_u16, v_b_factor);
    print_m256i(v_b_product);
    __m256i v_g_product = _mm256_mullo_epi16(v_g_u16, v_g_factor);
    print_m256i(v_g_product);
    __m256i v_r_product = _mm256_mullo_epi16(v_r_u16, v_r_factor);
    print_m256i(v_r_product);

    __m256i v_sum = _mm256_add_epi16(v_b_product, _mm256_add_epi16(v_g_product, v_r_product));
    print_m256i(v_sum);
    __m256i v_res_u16 = _mm256_srli_epi16(_mm256_add_epi16(v_sum, v_half), 8); // 右移八位相当于除3
    print_m256i(_mm256_add_epi16(v_sum, v_half));
    print_m256i(v_res_u16);
    __m256i v_res_u8 = _mm256_packus_epi16(v_res_u16, v_zero);
    print_m256i_u8(v_res_u8);
    __m256i v_res = _mm256_permute4x64_epi64(v_res_u8, split_mask); // 控制排列操作
    print_m256i_u8(v_res);

    _mm_storeu_si128((__m128i *)dst, _mm256_castsi256_si128(v_res)); //_mm256_castsi256_si128(v_res)控制低的128位向量.

    for (int i = 0; i < gray_data.size(); i++)
    {
        std::cout << static_cast<int>(gray_data[i]) << " ";
    }
    std::cout << std::endl;
    return 0;
}