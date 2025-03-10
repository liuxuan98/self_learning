#ifndef MAGIC_XE_CUDA_MATH_OPS_HPP_
#define MAGIC_XE_CUDA_MATH_OPS_HPP_
#include <cuda_runtime_api.h>

#include <algorithm>

#include "magic_xe_type_traits.hpp"

#define CUDA_TYPE(type, channel) type##channel
#define MAKE_CUDA_TYPE(type, channel) make_##type##channel

#define OPERATOR_VEC_FUNC_C1(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 1)(a.x OP b.x)
#define OPERATOR_VEC_FUNC_C2(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 2)(a.x OP b.x, a.y OP b.y)
#define OPERATOR_VEC_FUNC_C3(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 3)(a.x OP b.x, a.y OP b.y, a.z OP b.z)
#define OPERATOR_VEC_FUNC_C4(a, b, RES_TYPE, OP) \
    MAKE_CUDA_TYPE(RES_TYPE, 4)(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w)

#define OPERATOR_VEC_FUNC_IN_PLACE_C1(a, b, OP) a.x OP## = b.x;

#define OPERATOR_VEC_FUNC_IN_PLACE_C2(a, b, OP) \
    a.x OP## = b.x;                             \
    a.y OP## = b.y;

#define OPERATOR_VEC_FUNC_IN_PLACE_C3(a, b, OP) \
    a.x OP## = b.x;                             \
    a.y OP## = b.y;                             \
    a.z OP## = b.z;

#define OPERATOR_VEC_FUNC_IN_PLACE_C4(a, b, OP) \
    a.x OP## = b.x;                             \
    a.y OP## = b.y;                             \
    a.z OP## = b.z;                             \
    a.w OP## = b.w;

#define OPERATOR_VEC_VEC_FUNC(vec_type1, vec_type2, res_type, channel, op)                  \
    inline __host__ __device__ CUDA_TYPE(res_type, channel) operator op(                    \
        const CUDA_TYPE(vec_type1, channel) & a, const CUDA_TYPE(vec_type2, channel) & b) { \
        return OPERATOR_VEC_FUNC_C##channel(a, b, res_type, op);                            \
    }

#define OPERATOR_VEC_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, op) \
    OPERATOR_VEC_VEC_FUNC(vec_type1, vec_type2, res_type, 1, op)           \
    OPERATOR_VEC_VEC_FUNC(vec_type1, vec_type2, res_type, 2, op)           \
    OPERATOR_VEC_VEC_FUNC(vec_type1, vec_type2, res_type, 3, op)           \
    OPERATOR_VEC_VEC_FUNC(vec_type1, vec_type2, res_type, 4, op)

OPERATOR_VEC_VEC_FUNC_C1_TO_C4(int, int, int, %)

#define OPERATOR_VEC_FUNC(vec_type1, vec_type2, res_type, channel, op)                      \
    inline __host__ __device__ CUDA_TYPE(res_type, channel) operator op(                    \
        const CUDA_TYPE(vec_type1, channel) & a, const CUDA_TYPE(vec_type2, channel) & b) { \
        return OPERATOR_VEC_FUNC_C##channel(a, b, res_type, op);                            \
    }                                                                                       \
    inline __host__ __device__ CUDA_TYPE(vec_type1, channel) &operator op##=(               \
        CUDA_TYPE(vec_type1, channel) & a, const CUDA_TYPE(vec_type2, channel) & b) {       \
        OPERATOR_VEC_FUNC_IN_PLACE_C##channel(a, b, op) return a;                           \
    }

#define OPERATOR_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, op) \
    OPERATOR_VEC_FUNC(vec_type1, vec_type2, res_type, 1, op)           \
    OPERATOR_VEC_FUNC(vec_type1, vec_type2, res_type, 2, op)           \
    OPERATOR_VEC_FUNC(vec_type1, vec_type2, res_type, 3, op)           \
    OPERATOR_VEC_FUNC(vec_type1, vec_type2, res_type, 4, op)

#define OPERATOR_VEC_FUNC_OP(vec_type1, vec_type2, res_type)      \
    OPERATOR_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, +) \
    OPERATOR_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, -) \
    OPERATOR_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, *) \
    OPERATOR_VEC_FUNC_C1_TO_C4(vec_type1, vec_type2, res_type, /)

OPERATOR_VEC_FUNC_OP(uchar, uchar, uchar)
OPERATOR_VEC_FUNC_OP(float, float, float)
OPERATOR_VEC_FUNC_OP(uchar, float, float)
OPERATOR_VEC_FUNC_OP(float, uchar, float)
OPERATOR_VEC_FUNC_OP(int, int, int)

#define OPERATOR_SCA_VEC_FUNC_C1(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 1)(a OP b.x)
#define OPERATOR_SCA_VEC_FUNC_C2(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 2)(a OP b.x, a OP b.y)
#define OPERATOR_SCA_VEC_FUNC_C3(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 3)(a OP b.x, a OP b.y, a OP b.z)
#define OPERATOR_SCA_VEC_FUNC_C4(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 4)(a OP b.x, a OP b.y, a OP b.z, a OP b.w)

#define OPERATOR_VEC_SCA_FUNC_C1(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 1)(a.x OP b)
#define OPERATOR_VEC_SCA_FUNC_C2(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 2)(a.x OP b, a.y OP b)
#define OPERATOR_VEC_SCA_FUNC_C3(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 3)(a.x OP b, a.y OP b, a.z OP b)
#define OPERATOR_VEC_SCA_FUNC_C4(a, b, RES_TYPE, OP) MAKE_CUDA_TYPE(RES_TYPE, 4)(a.x OP b, a.y OP b, a.z OP b, a.w OP b)

#define OPERATOR_SCA_VEC_FUNC(scalar_type, vec_type, res_type, channel, op) \
    inline __host__ __device__ CUDA_TYPE(res_type, channel) operator op(    \
        const scalar_type &a, const CUDA_TYPE(vec_type, channel) & b) {     \
        return OPERATOR_SCA_VEC_FUNC_C##channel(a, b, res_type, op);        \
    }                                                                       \
    inline __host__ __device__ CUDA_TYPE(res_type, channel) operator op(    \
        const CUDA_TYPE(vec_type, channel) & a, const scalar_type &b) {     \
        return OPERATOR_VEC_SCA_FUNC_C##channel(a, b, res_type, op);        \
    }

#define OPERATOR_SCA_VEC_FUNC_C1_TO_C4(scalar_type, vec_type, res_type, op) \
    OPERATOR_SCA_VEC_FUNC(scalar_type, vec_type, res_type, 1, op)           \
    OPERATOR_SCA_VEC_FUNC(scalar_type, vec_type, res_type, 2, op)           \
    OPERATOR_SCA_VEC_FUNC(scalar_type, vec_type, res_type, 3, op)           \
    OPERATOR_SCA_VEC_FUNC(scalar_type, vec_type, res_type, 4, op)

#define OPERATOR_SCA_VEC_FUNC_OP(scalar_type, vec_type, res_type)      \
    OPERATOR_SCA_VEC_FUNC_C1_TO_C4(scalar_type, vec_type, res_type, +) \
    OPERATOR_SCA_VEC_FUNC_C1_TO_C4(scalar_type, vec_type, res_type, -) \
    OPERATOR_SCA_VEC_FUNC_C1_TO_C4(scalar_type, vec_type, res_type, *) \
    OPERATOR_SCA_VEC_FUNC_C1_TO_C4(scalar_type, vec_type, res_type, /)

OPERATOR_SCA_VEC_FUNC_OP(unsigned char, uchar, uchar)
OPERATOR_SCA_VEC_FUNC_OP(float, uchar, float)
OPERATOR_SCA_VEC_FUNC_OP(float, float, float)
OPERATOR_SCA_VEC_FUNC_OP(double, uchar, float)
OPERATOR_SCA_VEC_FUNC_OP(double, float, float)
OPERATOR_SCA_VEC_FUNC_OP(int, int, int)
OPERATOR_SCA_VEC_FUNC_OP(int, float, float)
OPERATOR_SCA_VEC_FUNC_OP(int, uchar, int)

inline int DivUp(int a, int b) {
    return (int)ceil((float)a / b);
}

#define OPERATOR_COMPARISON_FUNC(OP)                                                \
    template <typename T, typename U>                                               \
    inline __host__ __device__ bool operator OP(const T &a, const U &b) {           \
        using BASE_TYPE = BaseType<T>;                                              \
        for (int i = 0; i < NumElements<T>(); ++i) {                                \
            if (GetElement<BASE_TYPE, T>(a, i) OP GetElement<BASE_TYPE, U>(b, i)) { \
                return true;                                                        \
            }                                                                       \
        }                                                                           \
        return false;                                                               \
    }

OPERATOR_COMPARISON_FUNC(>)
OPERATOR_COMPARISON_FUNC(<)
OPERATOR_COMPARISON_FUNC(>=)
OPERATOR_COMPARISON_FUNC(<=)
OPERATOR_COMPARISON_FUNC(==)

template <typename T>
inline __host__ __device__ T abs(const T &a) {
    using BASE_TYPE = BaseType<T>;
    T res;
#pragma unroll
    for (int i = 0; i < NumElements<T>(); ++i) {
        GetElement<BASE_TYPE, T>(res, i) = std::abs(GetElement<BASE_TYPE, T>(a, i));
    }
    return res;
}

template <typename T>
inline __host__ __device__ T RoundAndAbs(const T &a) {
    using BASE_TYPE = BaseType<T>;
    T res;
#pragma unroll
    for (int i = 0; i < NumElements<T>(); ++i) {
        GetElement<BASE_TYPE, T>(res, i) = ::abs(::round(GetElement<BASE_TYPE, T>(a, i)));
    }
    return res;
}

template <typename T>
inline __host__ __device__ T round(const T &a) {
    using BASE_TYPE = BaseType<T>;
    T res;
#pragma unroll
    for (int i = 0; i < NumElements<T>(); ++i) {
        GetElement<BASE_TYPE, T>(res, i) = ::round(GetElement<BASE_TYPE, T>(a, i));
    }
    return res;
}

#endif