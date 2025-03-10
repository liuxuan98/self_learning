#ifndef MAGIC_XE_TYPE_TRAITS_HPP_
#define MAGIC_XE_TYPE_TRAITS_HPP_

#include <cuda_runtime_api.h>
#include <float.h>
#include <limits.h>

#ifndef uchar
typedef unsigned char uchar;
#endif

template <typename T>
struct TypeTraits;

#define MAGIC_XE_CUDA_TYPE_TRAITS(COMPOUND_TYPE, BASE_TYPE, COMPONENTS, ELEMENTS, MIN_VAL, MAX_VAL) \
    template <>                                                                                     \
    struct TypeTraits<COMPOUND_TYPE> {                                                              \
        using base_type = BASE_TYPE;                                                                \
        using value_type = COMPOUND_TYPE;                                                           \
        static constexpr int components = COMPONENTS;                                               \
        static constexpr int elements = ELEMENTS;                                                   \
        static constexpr base_type min = MIN_VAL;                                                   \
        static constexpr base_type max = MAX_VAL;                                                   \
    };
MAGIC_XE_CUDA_TYPE_TRAITS(unsigned char, unsigned char, 0, 1, 0, UCHAR_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS(int, int, 0, 1, INT_MIN, INT_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS(float, float, 0, 1, -FLT_MAX, FLT_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS(double, double, 0, 1, DBL_MIN, DBL_MAX)

#define MAGIC_XE_CUDA_TYPE_TRAITS_1_TO_4(COMPOUND_TYPE, BASE_TYPE, MIN_VAL, MAX_VAL) \
    MAGIC_XE_CUDA_TYPE_TRAITS(COMPOUND_TYPE##1, BASE_TYPE, 1, 1, MIN_VAL, MAX_VAL)   \
    MAGIC_XE_CUDA_TYPE_TRAITS(COMPOUND_TYPE##2, BASE_TYPE, 2, 2, MIN_VAL, MAX_VAL)   \
    MAGIC_XE_CUDA_TYPE_TRAITS(COMPOUND_TYPE##3, BASE_TYPE, 3, 3, MIN_VAL, MAX_VAL)   \
    MAGIC_XE_CUDA_TYPE_TRAITS(COMPOUND_TYPE##4, BASE_TYPE, 4, 4, MIN_VAL, MAX_VAL)

MAGIC_XE_CUDA_TYPE_TRAITS_1_TO_4(uchar, unsigned char, 0, UCHAR_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS_1_TO_4(int, int, INT_MIN, INT_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS_1_TO_4(float, float, -FLT_MAX, FLT_MAX)
MAGIC_XE_CUDA_TYPE_TRAITS_1_TO_4(double, double, DBL_MIN, DBL_MAX)

template <typename T>
__forceinline__ __host__ __device__ bool IsCompound() {
    return TypeTraits<T>::components >= 1;
}

template <typename T>
constexpr __forceinline__ __host__ __device__ int NumElements() {
    return TypeTraits<T>::elements;
}

template <typename T>
using BaseType = typename TypeTraits<T>::base_type;

template <typename T>
using ValueType = typename TypeTraits<T>::value_type;

template <typename T, typename U>
__host__ __device__ T &GetElement(const U &val, int idx) {
    if (IsCompound<U>()) {
        return ((T *)(&val))[idx];
    } else {
        return (T &)val;
    }
}

template <typename T, int NC>
struct MakeType;

#define MAGIC_XE_MAKE_TYPE(BASE_TYPE, NC, COMPOUND_TYPE) \
    template <>                                          \
    struct MakeType<BASE_TYPE, NC> {                     \
        using type = COMPOUND_TYPE;                      \
    }

#define MAGIC_XE_MAKE_TYPE_0_TO_4(BASE_TYPE, COMPOUND_TYPE) \
    MAGIC_XE_MAKE_TYPE(BASE_TYPE, 0, BASE_TYPE);            \
    MAGIC_XE_MAKE_TYPE(BASE_TYPE, 1, BASE_TYPE);            \
    MAGIC_XE_MAKE_TYPE(BASE_TYPE, 2, COMPOUND_TYPE##2);     \
    MAGIC_XE_MAKE_TYPE(BASE_TYPE, 3, COMPOUND_TYPE##3);     \
    MAGIC_XE_MAKE_TYPE(BASE_TYPE, 4, COMPOUND_TYPE##4);

MAGIC_XE_MAKE_TYPE_0_TO_4(uchar, uchar)
MAGIC_XE_MAKE_TYPE_0_TO_4(int, int)
MAGIC_XE_MAKE_TYPE_0_TO_4(float, float)
MAGIC_XE_MAKE_TYPE_0_TO_4(double, double)

template <typename T, int NC>
using MakeType_t = typename MakeType<T, NC>::type;

#endif