#ifndef MAGIC_XE_SATURATE_CAST_HPP_
#define MAGIC_XE_SATURATE_CAST_HPP_
#include <cuda_runtime_api.h>

#include "magic_xe_type_traits.hpp"

template <typename T, typename U>
__host__ __device__ T SaturateCastImpl(const U &v) {

    return (T)(v <= TypeTraits<T>::min ? TypeTraits<T>::min : v >= TypeTraits<T>::max ? TypeTraits<T>::max : v);
}

template <typename T, typename U>
__host__ __device__ T SaturateCast(const U &val) {
    using BU = BaseType<U>;
    using BT = BaseType<T>;

    T result;

    for (int i = 0; i < NumElements<T>(); ++i) {
        GetElement<BT, T>(result, i) = SaturateCastImpl<BT, BU>(GetElement<BU, U>(val, i));
    }

    return result;
}

#endif