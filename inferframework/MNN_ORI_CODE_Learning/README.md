# 源码编译事项 ，mnn新特性的利用和加速，transform结构加速以及cl
mnn在win上进行编译并不如意，因为官方提供的文档根本很难成功进行编译，各种bug报个不停

因此打算记录一下较简单的编译方式：

（1）去github下载mnn源码

git clone  https://github.com/alibaba/MNN
（2）打开vs2019专用的命令行窗口


 因为我是64位的系统，因此选择的x64 native tools command prompt for vs2019

（3）cd到mnn源码的路径里

（4）mkdir build

（5）cd build

（6）cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..

（7）nmake

使用方法：包含头文件和链接相关的库进行使用


