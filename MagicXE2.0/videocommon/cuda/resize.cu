inline __device__ void GetCubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3) {
    constexpr float A = -0.75f;

    w0 = ((A * (delta + 1) - 5 * A) * (delta + 1) + 8 * A) * (delta + 1) - 4 * A;
    w1 = ((A + 2) * delta - (A + 3)) * delta * delta + 1;
    w2 = ((A + 2) * (1 - delta) - (A + 3)) * (1 - delta) * (1 - delta) + 1;
    w3 = 1.f - w0 - w1 - w2;
}

template <typename T>
__global__ void resize_bilinear(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int height = src_size.y, width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;

    if ((dst_x < out_width) && (dst_y < out_height)) {
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int sy = static_cast<int>(fy);
        fy -= sy;
        sy = ::max(0, ::min(sy, height - 2));

        const T *src_ptr0 = (const T *)(src + sy * src_stride);
        const T *src_ptr1 = (const T *)(src + (sy + 1) * src_stride);
        T *dst_ptr = (T *)(dst + dst_y * dst_stride);

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int sx = static_cast<int>(fx);
        fx -= sx;
        sx = ::max(0, ::min(sx, width - 2));

        dst_ptr[dst_x] = SaturateCast<T>((1.f - fx) * ((1.f - fy) * src_ptr0[sx] + fy * src_ptr1[sx])
                                         + fx * ((1.f - fy) * src_ptr0[(sx + 1)] + fy * src_ptr1[(sx + 1)]));
    }
}

template <typename T>
__global__ void resize_cubic(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int height = src_size.y, width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;

    if ((dst_x < out_width) && (dst_y < out_height)) {
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int sy = static_cast<int>(::floor(fy));
        fy -= sy;
        sy = ::max(1, ::min(sy, height - 2));

        //const T *src_ptr0 = (const T *)(src + (sy - 1) * src_stride);
        // const T *src_ptr1 = (const T *)(src + sy * src_stride); //four row point
        // const T *src_ptr2 = (const T *)(src + (sy + 1) * src_stride);
        // const T *src_ptr3 = (const T *)(src + (sy + 2) * src_stride);
        T *dst_ptr = (T *)(dst + dst_y * dst_stride);

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int sx = static_cast<int>(::floor(fx));
        fx -= sx;

        fx = (sx < 1 || sx >= width - 2) ? 0 : fx;
        sx = ::max(1, ::min(sx, width - 2));

        float wx[4];
        float wy[4];

        GetCubicCoeffs(fx, wx[0], wx[1], wx[2], wx[3]); //weights_x.
        GetCubicCoeffs(fy, wy[0], wy[1], wy[2], wy[3]); //weights_y.

        MakeType_t<float, NumElements<T>()> sum = {};

#pragma unroll
        for (int cy = -1; cy <= 2; cy++) {
            const T *src_ptr = (const T *)(src + (sy + cy) * src_stride);
#pragma unroll
            for (int cx = -1; cx <= 2; cx++) {
                sum = sum + (src_ptr[sx + cx] * (wx[cx + 1] * wy[cy + 1]));
            }
        }
        dst_ptr[dst_x] = SaturateCast<T>(RoundAndAbs(sum));
    }
}

template <typename T>
__global__ void resize_nn(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int height = src_size.y, width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;

    if ((dst_x < out_width) && (dst_y < out_height)) {
        int src_x = ::min((int)::round(dst_x * scale_x), src_size.x - 1);
        int src_y = ::min((int)::round(dst_y * scale_y), src_size.y - 1);

        const T *src_ptr = (const T *)(src + src_y * src_stride);
        T *dst_ptr = (T *)(dst + dst_y * dst_stride);

        dst_ptr[dst_x] = SaturateCast<T>(src_ptr[src_x]);
    }
}

template <typename T>
__global__ void resize_area_zoomin(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int src_height = src_size.y, src_width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;

    if ((dst_x < out_width) && (dst_y < out_height)) {

        float inv_scale_x = 1.f / scale_x;
        float inv_scale_y = 1.f / scale_y;

        int sy = ::floor(dst_y * scale_y);
        float fy = (float)((dst_y + 1) - (sy + 1) * inv_scale_y);
        fy = fy <= 0 ? 0.f : fy - ::floor(fy);
        sy = ::min(sy, src_height - 1);

        float cbufy[2];
        cbufy[0] = 1.f - fy;
        cbufy[1] = 1.f - cbufy[0];

        int sx = ::floor(dst_x * scale_x);
        float fx = (float)((dst_x + 1) - (sx + 1) * inv_scale_x);
        fx = fx < 0 ? 0.f : fx - ::floor(fx);

        fx = (sx < 0 || sx >= src_width - 1) ? 0 : fx;
        sx = (sx < 0) ? 0 : sx;
        sx = (sx >= src_width - 1) ? src_width - 1 : sx;

        float cbufx[2];
        cbufx[0] = 1.f - fx;
        cbufx[1] = 1.f - cbufx[0];

        const T *src_ptr0 = (const T *)(src + sy * src_stride);
        const T *src_ptr1 = (const T *)(src + (sy + 1) * src_stride);

        T *dst_ptr = (T *)(dst + dst_y * dst_stride);
        dst_ptr[dst_x] = SaturateCast<T>(
            round(((cbufx[0] * cbufy[0]) * src_ptr0[sx] + (cbufx[0] * cbufy[1]) * src_ptr1[sx]
                   + (cbufx[1] * cbufy[0]) * src_ptr0[(sx + 1)] + (cbufx[1] * cbufy[1]) * src_ptr1[(sx + 1)])));
    }
}

template <typename T>
__global__ void resize_area_intzoomout(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int src_height = src_size.y, src_width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;

    if ((dst_x < out_width) && (dst_y < out_height)) {

        //zoom out dst height an weight compress.
        // integer multiples compress.
        int sy = ::min((int)::floor(dst_y * scale_y), src_height - 1);
        int sx = ::min((int)::floor(dst_x * scale_x), src_width - 1);

        T *dst_ptr = (T *)(dst + dst_y * dst_stride);

        MakeType_t<float, NumElements<T>()> sum = {};
#pragma unroll
        for (int i = 0; i < (int)scale_y; ++i) {
            const T *src_ptr = (const T *)(src + (sy + i) * src_stride);
#pragma unroll
            for (int j = 0; j < (int)scale_x; ++j) {
                sum += src_ptr[sx + j];
            }
        }

        dst_ptr[dst_x] = SaturateCast<T>(round(sum / float(scale_x * scale_y))); //round

        return;
    }
}

template <typename T>
__global__ void resize_area_floatzoomout(const unsigned char *src,
    const int src_stride,
    unsigned char *dst,
    const int dst_stride,
    int2 src_size,
    int2 dst_size,
    const float scale_x,
    const float scale_y) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int src_height = src_size.y, src_width = src_size.x, out_height = dst_size.y, out_width = dst_size.x;
    if ((dst_x < out_width) && (dst_y < out_height)) {
        //no integer multiples compress.
        float fsy1 = dst_y * scale_y; //begin
        float fsy2 = fsy1 + scale_y;  //end
        float cell_height = ::min(scale_y, src_height - fsy1);

        int sy1 = ::ceil(fsy1), sy2 = ::floor(fsy2);

        sy2 = ::min(sy2, src_height - 1);
        sy1 = ::min(sy1, sy2);

        float fsx1 = dst_x * scale_x;
        float fsx2 = fsx1 + scale_x;
        float cell_width = ::min(scale_x, src_width - fsx1);

        int sx1 = ::ceil(fsx1), sx2 = ::floor(fsx2);
        sx2 = ::min(sx2, src_width - 1);
        sx1 = ::min(sx1, sx2);

        MakeType_t<float, NumElements<T>()> region_sum = {0};
        //#pragma unroll
        if (sy1 - fsy1 > 1e-3) {
            const T *src_ptr = (const T *)(src + (sy1 - 1) * src_stride);
            float weight_y = ((float)sy1 - fsy1) / cell_height;
            float weight_x = 0.f;
            if (sx1 - fsx1 > 1e-3) {
                weight_x = ((float)sx1 - fsx1) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx1 - 1];
            }
#pragma unroll
            for (int sx = sx1; sx < sx2; sx++) {
                weight_x = 1.f / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx];
            }

            if (fsx2 - sx2 > 1e-3) {
                weight_x = ::min(::min(fsx2 - sx2, 1.0f), cell_width) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx2];
            }
        }
#pragma unroll
        for (int sy = sy1; sy < sy2; ++sy) {
            const T *src_ptr = (const T *)(src + sy * src_stride);
            float weight_y = 1.f / cell_height;
            float weight_x = 0;
            if (sx1 - fsx1 > 1e-3) {
                weight_x = ((float)sx1 - fsx1) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx1 - 1];
            }
#pragma unroll
            for (int sx = sx1; sx < sx2; sx++) {
                weight_x = 1.f / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx];
            }

            if (fsx2 - sx2 > 1e-3) {
                weight_x = ::min(::min(fsx2 - sx2, 1.0f), cell_width) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx2];
            }
        }
        //#pragma unroll
        if (fsy2 - sy2 > 1e-3) {
            const T *src_ptr = (const T *)(src + sy2 * src_stride);
            float weight_y = (float)::min(::min(fsy2 - sy2, 1.0f), cell_height) / cell_height;
            float weight_x = 0;

            if (sx1 - fsx1 > 1e-3) {
                weight_x = ((float)sx1 - fsx1) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx1 - 1];
            }
#pragma unroll
            for (int sx = sx1; sx < sx2; sx++) {
                weight_x = 1.f / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx];
            }

            if (fsx2 - sx2 > 1e-3) {
                weight_x = ::min(::min(fsx2 - sx2, 1.0f), cell_width) / cell_width;
                region_sum += (weight_y * weight_x) * src_ptr[sx2];
            }
        }

        T *dst_ptr = (T *)(dst + dst_y * dst_stride);
        dst_ptr[dst_x] = SaturateCast<T>(round(region_sum));
    }
}

template <typename T>
void Resize(const MagicXEFrameV2 *src_frame, MagicXEFrameV2 *des_frame, MagicXEInterpolationFlags sample_type) {
    int2 src_size{src_frame->video.width, src_frame->video.height};
    int2 dst_size{des_frame->video.width, des_frame->video.height};

    float scale_x = static_cast<float>(src_size.x) / dst_size.x;
    float scale_y = static_cast<float>(src_size.y) / dst_size.y;

    const int threads_per_block = 256;
    const int block_width = 8;

    const dim3 block_size(block_width, threads_per_block / block_width, 1);
    const dim3 grid_size(DivUp(dst_size.x, block_size.x), DivUp(dst_size.y, block_size.y), 1);

    const MagicXEEnv *env = MagicXEEnvGet();
    cudaStream_t stream = (cudaStream_t)(env->env_dev.cuda_stream);

    if (sample_type == XE_INTER_LINEAR) {
        resize_bilinear<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
            src_frame->linesize[0],
            des_frame->data[0],
            des_frame->linesize[0],
            src_size,
            dst_size,
            scale_x,
            scale_y);
    } else if (sample_type == XE_INTER_NEAREST) {
        resize_nn<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
            src_frame->linesize[0],
            des_frame->data[0],
            des_frame->linesize[0],
            src_size,
            dst_size,
            scale_x,
            scale_y);
    } else if (sample_type == XE_INTER_CUBIC) {
        resize_cubic<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
            src_frame->linesize[0],
            des_frame->data[0],
            des_frame->linesize[0],
            src_size,
            dst_size,
            scale_x,
            scale_y);
    } else if (sample_type == XE_INTER_AREA) {
        int iscale_x = static_cast<int>(scale_x);
        int iscale_y = static_cast<int>(scale_y);
        bool is_erea_fast = ::abs(scale_x - iscale_x) < DBL_EPSILON && ::abs(scale_y - iscale_y) < DBL_EPSILON;
        if (scale_x >= 1 && scale_y >= 1) {
            if (is_erea_fast) {

                resize_area_intzoomout<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
                    src_frame->linesize[0],
                    des_frame->data[0],
                    des_frame->linesize[0],
                    src_size,
                    dst_size,
                    scale_x,
                    scale_y);
            } else {
                resize_area_floatzoomout<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
                    src_frame->linesize[0],
                    des_frame->data[0],
                    des_frame->linesize[0],
                    src_size,
                    dst_size,
                    scale_x,
                    scale_y);
            }

        } else {
            resize_area_zoomin<T><<<grid_size, block_size, 0, stream>>>(src_frame->data[0],
                src_frame->linesize[0],
                des_frame->data[0],
                des_frame->linesize[0],
                src_size,
                dst_size,
                scale_x,
                scale_y);
        }

    } else {
        MAGIC_XE_LOGE("Resize unsupport sample type:%d", sample_type);
        return;
    }

    return;
}

MagicXEError MagicXECudaResize(
    const MagicXEFrameV2 *src_frame, MagicXEFrameV2 *des_frame, MagicXEInterpolationFlags sample_type) {
    if (src_frame == NULL || des_frame == NULL) {
        MAGIC_XE_LOGE("MagicXEFrameResizeCUDA src frame or dst frame is NULL\n");
        return MAGIC_XE_SUCCESS;
    }

    MagicXECudaDataFmt data_fmt = GetDataFmt(src_frame->video.format);
    if (data_fmt == CudaDataFmt_None) {
        MAGIC_XE_LOGE("MagicXEFrameResizeCUDA unsupport data format:%d", data_fmt);
        return MAGIC_XE_INVALID_PARAM;
    }

    typedef void (*resize_func)(
        const MagicXEFrameV2 *src_frame, MagicXEFrameV2 *des_frame, MagicXEInterpolationFlags sample_type);

    static resize_func func_tab[2][4] = {{Resize<unsigned char>, Resize<uchar2>, Resize<uchar3>, Resize<uchar4>},
        {Resize<float>, Resize<float2>, Resize<float3>, Resize<float4>}};

    resize_func func = func_tab[data_fmt][src_frame->video.channels - 1];
    if (func == nullptr) {
        MAGIC_XE_LOGE(
            "MagicXEFrameResizeCUDA unsupport data format:%d, channels:%d", data_fmt, src_frame->video.channels);
        return MAGIC_XE_INVALID_PARAM;
    }

    func(src_frame, des_frame, sample_type);

    return MAGIC_XE_SUCCESS;
}