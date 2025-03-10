#include <math.h>
#include <stdint.h>



static void InvertMat(const float *m, float *coeffs) {
    float den = m[0] * m[4] - m[1] * m[3];
    den = std::abs(den) > 1e-5 ? 1. / den : .0;
    coeffs[0] = m[4] * den;
    coeffs[1] = -m[1] * den;
    coeffs[2] = (m[1] * m[5] - m[2] * m[4]) * den;
    coeffs[3] = -m[3] * den;
    coeffs[4] = m[0] * den;
    coeffs[5] = (m[2] * m[3] - m[0] * m[5]) * den;
}

struct WarpAffineTransform {
    static __host__ __device__ float2 CalcCoord(const float *warp_matrix, int x, int y) {
        const float x_coord = warp_matrix[0] * x + warp_matrix[1] * y + warp_matrix[2];
        const float y_coord = warp_matrix[3] * x + warp_matrix[4] * y + warp_matrix[5];
        return make_float2(x_coord, y_coord);
    }

    float xform[9];
};

template <typename T, MagicXEBorderType B>
class BorderWrap {
public:
    BorderWrap() {
    }

    BorderWrap(const uint8_t *src, int2 src_size, int stride, double constant_val)
        : src_(src), src_size_(src_size), stride_(stride), constant_val_(constant_val) {
    }

    inline __host__ __device__ T &operator[](int2 c) {
        val_ = SaturateCast<T>(0);
        int2 coord = make_int2(GetIndexWithBorder<B>(c.x, src_size_.x), GetIndexWithBorder<B>(c.y, src_size_.y));

        const T *src_ptr = (const T *)(src_ + coord.y * stride_);
        val_ = SaturateCast<T>(src_ptr[coord.x]);

        return (T &)val_;
    }

private:
    const uint8_t *src_;
    int2 src_size_;
    int stride_;
    double constant_val_;

    T val_;
};

template <typename T>
class BorderWrap<T, XE_BORDER_CONSTANT> {
public:
    BorderWrap() {
    }

    BorderWrap(const uint8_t *src, int2 src_size, int stride, double constant_val)
        : src_(src), src_size_(src_size), stride_(stride), constant_val_(constant_val) {
    }

    inline __host__ __device__ T &operator[](int2 c) {
        if (c.x < 0 || c.x >= src_size_.x || c.y < 0 || c.y >= src_size_.y) {
            val_ = SaturateCast<T>(constant_val_);
        } else {
            const T *src_ptr = (const T *)(src_ + c.y * stride_);
            val_ = SaturateCast<T>(src_ptr[c.x]);
        }

        return (T &)val_;
    }

private:
    const uint8_t *src_;
    int2 src_size_;
    int stride_;
    double constant_val_;

    T val_;
};

template <typename T, class B, MagicXEInterpolationFlags I>
class InterpolationWarp;

template <typename T, class B>
class InterpolationWarp<T, B, XE_INTER_NEAREST> {
public:
    InterpolationWarp() {
    }

    InterpolationWarp(B border_wrap) : border_wrap_(border_wrap) {
    }

    inline __host__ __device__ T &operator[](float2 c) {
        int x_coord = (int)(c.x + 0.5f);
        int y_coord = (int)(c.y + 0.5f);
        int2 coord = make_int2(x_coord, y_coord);

        return (T &)border_wrap_[coord];
    }

private:
    B border_wrap_;
};

template <typename T, class B>
class InterpolationWarp<T, B, XE_INTER_LINEAR> {
public:
    InterpolationWarp() {
    }

    InterpolationWarp(B border_wrap) : border_wrap_(border_wrap) {
    }

    inline __host__ __device__ T operator[](float2 c) {
        int x1 = (int)(c.x);
        int y1 = (int)(c.y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        using ValueType = MakeType_t<float, NumElements<T>()>;

        ValueType val = border_wrap_[{x1, y1}] * (x2 - c.x) * (y2 - c.y);
        val += border_wrap_[{x2, y1}] * (c.x - x1) * (y2 - c.y);
        val += border_wrap_[{x1, y2}] * (x2 - c.x) * (c.y - y1);
        val += border_wrap_[{x2, y2}] * (c.x - x1) * (c.y - y1);

        return SaturateCast<T>(val);
    }

private:
    B border_wrap_;
    T val_;
};

template <typename T, class SrcWrapper>
__global__ void WarpAffineKernel(
    SrcWrapper src, uint8_t *dst, int2 dst_size, int dst_stride, WarpAffineTransform transform) {
    
//         模板参数：

// T：表示像素数据类型（例如 uchar, float 等）。
// SrcWrapper：一个包装类，用于处理边界填充和插值等操作。
// 参数：

// SrcWrapper src：源图像的数据包装器。
// uint8_t *dst：指向目标图像数据的指针。
// int2 dst_size：目标图像的尺寸（宽度和高度）。
// int dst_stride：目标图像每一行的字节数（步长）。
// WarpAffineTransform transform：包含仿射变换矩阵的对象。
    //2. 计算线程索引
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    //3. 共享内存初始化
    extern __shared__ float coeffs[9];
    if (tid < 9) {
        coeffs[tid] = transform.xform[tid];
    }

    __syncthreads();
    //4. 边界检查和坐标计算
    if (dst_x < dst_size.x && dst_y < dst_size.y) {
        float2 src_coord = WarpAffineTransform::CalcCoord(coeffs, dst_x, dst_y);
        //5. 数据写入目标图像
        T *dst_ptr = (T *)(dst + dst_y * dst_stride);
        dst_ptr[dst_x] = SaturateCast<T>(src[src_coord]);
    }
}

template <typename T, MagicXEInterpolationFlags I, MagicXEBorderType B>
struct WarpAffineDispatcher {
    static void Call(MagicXEFrameV2 *dst_frame,
        const MagicXEFrameV2 *src_frame,
        WarpAffineTransform transform,
        double constant_val) {

        BorderWrap<T, B> border_wrap(src_frame->data[0],
            make_int2(src_frame->video.width, src_frame->video.height),
            src_frame->linesize[0],
            constant_val);
        InterpolationWarp<T, BorderWrap<T, B>, I> src_wrapper(border_wrap);

        int2 dst_size = {dst_frame->video.width, dst_frame->video.height};

        const int thread_per_block = 256;
        const int block_width = 8;
        const dim3 block_size(block_width, thread_per_block / block_width, 1);
        const dim3 grid_size(DivUp(dst_size.x, block_size.x), DivUp(dst_size.y, block_size.y), 1);

        const int shr_mem = 9 * sizeof(float);

        const MagicXEEnv *env = MagicXEEnvGet();
        cudaStream_t stream = (cudaStream_t)(env->env_dev.cuda_stream);

        WarpAffineKernel<T><<<grid_size, block_size, shr_mem, stream>>>(
            src_wrapper, dst_frame->data[0], dst_size, dst_frame->linesize[0], transform);
    }
};

template <typename T>
void WarpAffine(MagicXEFrameV2 *dst_frame,
    const MagicXEFrameV2 *src_frame,
    WarpAffineTransform transform,
    MagicXEInterpolationFlags interpolation,
    MagicXEBorderType border_type,
    double constant_val) {

    typedef void (*warp_affine_dispatcher)(MagicXEFrameV2 *, const MagicXEFrameV2 *, WarpAffineTransform, double);

    static warp_affine_dispatcher func_tab[2][5] = {
        {WarpAffineDispatcher<T, XE_INTER_NEAREST, XE_BORDER_CONSTANT>::Call,
            WarpAffineDispatcher<T, XE_INTER_NEAREST, XE_BORDER_REPLICATE>::Call,
            WarpAffineDispatcher<T, XE_INTER_NEAREST, XE_BORDER_REFLECT>::Call,
            WarpAffineDispatcher<T, XE_INTER_NEAREST, XE_BORDER_WRAP>::Call,
            WarpAffineDispatcher<T, XE_INTER_NEAREST, XE_BORDER_REFLECT_101>::Call},
        {WarpAffineDispatcher<T, XE_INTER_LINEAR, XE_BORDER_CONSTANT>::Call,
            WarpAffineDispatcher<T, XE_INTER_LINEAR, XE_BORDER_REPLICATE>::Call,
            WarpAffineDispatcher<T, XE_INTER_LINEAR, XE_BORDER_REFLECT>::Call,
            WarpAffineDispatcher<T, XE_INTER_LINEAR, XE_BORDER_WRAP>::Call,
            WarpAffineDispatcher<T, XE_INTER_LINEAR, XE_BORDER_REFLECT_101>::Call}};

    func_tab[interpolation][border_type](dst_frame, src_frame, transform, constant_val);
}

MagicXEError MagicXECUDAWarpAffine(MagicXEFrameV2 *dst_frame,
    const MagicXEFrameV2 *src_frame,
    const float *trans_matrix,
    MagicXEInterpolationFlags interpolation,
    MagicXEBorderType border_type,
    double constant_val) {
    MagicXEError ret = MAGIC_XE_SUCCESS;

    MagicXECudaDataFmt data_fmt = GetDataFmt(src_frame->video.format);

    WarpAffineTransform transform;
    InvertMat(trans_matrix, transform.xform);

    typedef void (*warp_affine_func)(MagicXEFrameV2 *,
        const MagicXEFrameV2 *,
        WarpAffineTransform,
        MagicXEInterpolationFlags,
        MagicXEBorderType,
        double);

    static warp_affine_func func_tab[2][4] = {
        {WarpAffine<uchar>, WarpAffine<uchar2>, WarpAffine<uchar3>, WarpAffine<uchar4>},
        {WarpAffine<float>, WarpAffine<float2>, WarpAffine<float3>, WarpAffine<float4>},
    };

    func_tab[data_fmt][src_frame->video.channels - 1](
        dst_frame, src_frame, transform, interpolation, border_type, constant_val);

    return ret;
}