#include "immintrin.h" //AVX(include wmmintrin.h)包含之前兼容的指令集的头文件
#include <vector>      //VC引入<intrin.h>会自动引入当前编译器所支持的所有Intrinsic头文件。GCC引入<x86intrin.h>.
#include <iostream>
#include <cstdint>

static inline void SwapElementU8(uint8_t *data, int idx_a, int idx_b)
{
    uint8_t a = data[idx_a], b = data[idx_b];
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
    data[idx_a] = a;
    data[idx_b] = b;
}
void print_m256i(__m256i vec)
{
    uint8_t arr[32];
    _mm256_storeu_si256((__m256i *)arr, vec);

    for (int i = 0; i < 32; ++i)
    {
        std::cout << static_cast<int>(arr[i]) << " ";
        if ((i + 1) % 8 == 0)
            std::cout << "| "; // 分隔每个32位边界，便于阅读
    }
    std::cout << std::endl;
}
template <typename T>
void gen_sequence(int n, std::vector<T> &dst)
{
    for (int index = 0; index < n; ++index)
    {
        dst[index] = static_cast<T>(index);
    }
}

void printf_vec()
{
}

// 编译命令cl /EHsc /O2 /arch:AVX2 avx_test.cc
int main()
{
    // 12 * 8
    // 默认支持avx.
    int n = 96; // 单次的交换算法avx指令集下的
    std::vector<uint8_t> src_vec(n);
    gen_sequence(n, src_vec);
    // printf("%d %d %d %d\n", src_vec1[0], src_vec1[1], src_vec1[2], src_vec1[3]);
    // std::vector<uint8_t> src_vec2(src_vec1); 调用拷贝构造函数

    const uint8_t *src = src_vec.data();
    // 32 *8 =256
    __m256i v_mask0 = _mm256_set_epi8(
        15, 14, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1, 0, 15, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2);
    __m256i v_mask1 = _mm256_set_epi8(
        15, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, 13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0); // 首位不动，后接壤内容最后一位不动
    __m256i v_mask2 = _mm256_set_epi8(
        13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0, 15, 14, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1, 0); // 前两位不动，

    // print_m256i(v_mask0);                                      // v_mask0 是从高位到低位的排序.
    __m256i v_src0 = _mm256_loadu_si256((const __m256i *)src);
    __m256i v_src1 = _mm256_loadu_si256((const __m256i *)(src + 32));
    __m256i v_src2 = _mm256_loadu_si256((const __m256i *)(src + 64));

    // print_m256i(v_src0);

    __m256i v_dst0 = _mm256_shuffle_epi8(v_src0, v_mask0);
    __m256i v_dst1 = _mm256_shuffle_epi8(v_src1, v_mask1);
    __m256i v_dst2 = _mm256_shuffle_epi8(v_src2, v_mask2);

    // __m256i v_src0 = _mm256_loadu_si256((const __m256i *)src); // 不保证严格的对齐.
    print_m256i(v_src0);
    print_m256i(v_dst0);
    print_m256i(v_src1);
    print_m256i(v_dst1);
    print_m256i(v_src2);
    print_m256i(v_dst2);

    //__m256i v_dst0 = _mm256_shuffle_epi8(v_src0, v_mask0);
    // print_m256i(v_dst0);
    std::vector<uint8_t> dst_vec(n, 0); // 内部寄存器地址没有报边界错误
    uint8_t *dst = dst_vec.data();
    //细节,字节位数问题 256位向量
    _mm256_storeu_si256((__m256i *)dst, v_dst0);
    _mm256_storeu_si256((__m256i *)(dst + 32), v_dst1);
    _mm256_storeu_si256((__m256i *)(dst + 64), v_dst2);
    printf("%d %d %d %d\n", src_vec[15], src_vec[17], src_vec[30], src_vec[32]);
    printf("%d %d %d %d\n", dst_vec[15], dst_vec[17], dst_vec[30], dst_vec[32]);

    printf("%d %d %d %d\n", src_vec[63], src_vec[65], src_vec[78], src_vec[80]);
    printf("%d %d %d %d\n", dst_vec[63], dst_vec[65], dst_vec[78], dst_vec[80]);

    // 置换元素
    SwapElementU8(dst, 15, 17);
    SwapElementU8(dst, 30, 32);
    SwapElementU8(dst, 63, 65);
    SwapElementU8(dst, 78, 80);

    for (size_t i = 0; i < dst_vec.size(); i++)
    {
        std::cout << static_cast<int>(dst_vec[i]) << " ";
    }
    std::cout << std::endl;

    // SwapElementU8(dst, 15, 17);
    return 0;
}