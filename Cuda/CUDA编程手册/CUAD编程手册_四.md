# 如何使用其它CUDA工具（TOOL-KIT）库进行编程

## tool-kits sdk 
    CUDA 工具包提供了许多线性代数、图像和信号处理以及随机处理的 GPU 加速库。它们包括 cuBLAS（基本线性代数子程序）、cuFFT（快速傅里叶变换）、cuRAND（随机数生成）、NPP（图像和信号处理）、cuSPARSE（稀疏线性代数）、nvGRAPH（图分析）、cuSolver（GPU 中的 LAPACK）、Thrust（CUDA 中的 STL）等。我们还可以使用 OpenCV 库编写 GPU 加速程序。

## 使用 cuBLAS 进行线性代数运算
    cuBLAS 库是 GPU 优化的基本线性代数子程序（BLAS）的标准实现。使用其 API，程序员可以将计算密集型代码优化为单个 GPU 或多个 GPU。cuBLAS 有三个级别。级别 1 执行矢量-矢量运算，级别 2 执行矩阵-矢量运算，级别 3 执行矩阵-矩阵运算。

    SGEMM运算，矩阵乘法运算，cuBLAS库的cuBLAS-XT API在多个GPU上运行时提供了cuBLAS的三级操作。

    混合精度的GEMM,使用TensorCores方法进行性能提升



https://github.com/PacktPublishing/Learn-CUDA-Programming/tree/master/