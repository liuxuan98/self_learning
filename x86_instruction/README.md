windows编译,编译器：MSVC.
@brief cl /EHsc /O2 /arch:AVX2 avx_RGB2Gary.cc SIMD（单指令多数据）体系结构，即向量化.
@return


https://www.cnblogs.com/ThousandPine/p/16964553.html 基础学习
SSE与AVX指令基础介绍与使用
SSE/AVX指令属于Intrinsics函数 
1.SSE/AVX指令属于Intrinsics函数，由编译器在编译时直接在调用处插入代码，避免了函数调用的额外开销。但又与inline函数不同，Intrinsics函数的代码由编译器提供，能够更高效地使用机器指令进行优化调整。

2.头文件
SSE和AVX指令集有多个不同版本，其函数也包含在对应版本的头文件里。

若不关心具体版本则可以使用 <intrin.h> 包含所有版本的头文件内容。
    '''
    #include <intrin.h> 
    '''
3.编译选项
除了头文件以外，我们还需要添加额外的编译选项，才能保证代码被编译成功。

各版本的SSE和AVX都有单独的编译选项，比如-msseN, -mavxN(N表示版本编号)。

经过简单测试后发现，此类编译选项支持向下兼容，比如-msse4可以编译SSE2的函数，-mavx也可以兼容各版本的SSE。

本文中的内容最多涉及到AVX2主要是我的CPU最多支持到这(悲)，所以只需要一个-mavx2就能正常运行文中的所有代码。

4.数据类型
Intel目前主要的SIMD指令集有MMX, SSE, AVX, AVX-512，其对处理的数据位宽分别是：

    ·64位 MMX
    ·128位 SSE
    ·256位 AVX
    ·512位 AVX-512
每种位宽对应一个数据类型，名称包括三个部分：

前缀 __m，两个下划线加m。
中间是数据位宽。
最后加上的字母表示数据类型，i为整数，d为双精度浮点数，不加字母则是单精度浮点数。
比如SSE指令集的 __m128, __m128i, __m128d

AVX则包括 __m256, __m256i, __m256d。

这里的位宽指的是SIMD寄存器的位宽，CPU需要先将数据加载进专门的寄存器之后再并行计算。

5.Intrinsic函数命名
同样，Intrinsic函数的命名通常也是由3个部分构成：
    5.1第一部分为前缀_mm，MMX和SSE都为_mm开头，AVX和AVX-512则会额外加上256和512的位宽标识。
    5.2第二部分表示执行的操作，比如_add,_mul,_load等，操作本身也会有一些修饰，比如_loadu表示以无需内存对齐的方式加载数据。
    5.3第三部分为操作选择的数据范围和数据类型，比如_ps的p(packed)表示所有数据，s(single)表示单精度浮点。_ss则表示s(single)第一个，s(single)单精度浮点。_epixx（xx为位宽）操作所有的xx位的有符号整数，_epuxx则是操作所有的xx位的无符号整数。
例如_mm256_load_ps表示将浮点数加载进整个256位寄存器中。

绝大部分Intrinsic函数都是按照这样的格式构成，每个函数也都能在Intel® Intrinsics Guide找到更为完整的描述。

6.SSE基础应用
内存对齐在指令集上特别重要对于性能优化也很重要
类型转换