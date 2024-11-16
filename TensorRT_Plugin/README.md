//todo
首先明确要开发的算子，最好是先用 CUDA 实现；
开发插件类，实现 IPluginV2DynamicExt 接口，在这里调用前面实现的算子；
开发插件 Creator 类，实现 IPluginCreator 接口，用于创建插件实例，然后注册该 Creator 类；
编译插件项目，生成动态链接库；
在构造 engine 之前，首先加载上一步编译出来的插件动态链接库，在构造 engine 时 TensorRT 会自动找到先前注册的插件创建器。

https://seanwangjs.github.io/2023/11/27/trt03-plugin.html 学习链接
https://developer.nvidia.com/zh-cn/blog/tensorrt-custom-layer-cn/ 学习链接