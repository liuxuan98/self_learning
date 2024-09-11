一、常见编译构建工具简单科普cmake+make
1.cmake: 是 CMake 的命令行工具，用于生成构建文件Makefile等文件.大型项目的构建CMakeList.txt
2.make(自动化构建工具)来编译Makefile(构建和管理项目的文件).
3.Makefile保存了编译器和连接器的参数选项，还表述了所有源文件之间(源文件和目标文件之间的依赖关系)的关系,确保编译时只编译需要重新编译的文件,提高编译效率.
4.cmake生成ninjia cmake -G ninjia ,ninjia(速度快)和Makefile都是
5.Makefile 的优点包括：

自动化构建: Makefile 可以自动化构建项目，节省时间和精力。
灵活性: Makefile 可以根据项目的需求进行定制。
可移植性: Makefile 可以在不同的平台上使用。
Makefile 的常用命令包括：

make: 执行 Makefile 中的规则和命令。
make all: 执行 Makefile 中的所有规则和命令。
make clean: 删除生成的文件和目标文件。
make install: 安装生成的文件和目标文件。

二、cmake -G: 是 CMake 的选项，用于指定生成器（Generator）。
这些都是 CMake 的生成器（Generator），用于生成不同类型的构建文件。下面是它们的差别：

MinGW Makefiles:
生成 MinGW 的 Makefile 文件。
适用于 MinGW 编译器。
使用 mingw32-make 命令构建项目。
主要用于 Windows 平台。
Visual Studio 15 2017 Win64:
生成 Visual Studio 2017 的项目文件（.vcxproj 文件）。
适用于 Visual Studio 2017 编译器。背后也是cl.exe微软编译器
使用 Visual Studio 2017 IDE 构建项目。
主要用于 Windows 平台。
NMake Makefiles: 
生成 NMake 的 Makefile 文件。
适用于 Microsoft 编译器（cl.exe）。
使用 nmake 命令构建项目。
主要用于 Windows 平台。
主要差别如下：

编译器: MinGW Makefiles 适用于 MinGW 编译器，Visual Studio 15 2017 Win64 适用于 Visual Studio 2017 编译器，NMake Makefiles 适用于 Microsoft 编译器。
构建工具: MinGW Makefiles 使用 mingw32-make 命令构建项目，Visual Studio 15 2017 Win64 使用 Visual Studio 2017 IDE 构建项目，NMake Makefiles 使用 nmake 命令构建项目。
平台: 所有三个生成器都主要用于 Windows 平台。
选择哪个生成器取决于你的项目需求和编译器选择。如果你使用 MinGW 编译器，可以选择 MinGW Makefiles。如果你使用 Visual Studio 2017，可以选择 Visual Studio 15 2017 Win64。如果你使用 Microsoft 编译器，可以选择 NMake Makefiles。

三、常见大型项目的构建和编译步骤
BUILD
cd /Users/ws/Downloads/MagicXE_Darwin_arm64_1.3.0.0/sample rm -rf build mkdir build && cd build

方法一
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DMAGIC_XE_ROOT_DIR=\Users\yanquan\Desktop\code\MagicXE_Windows_AMD64_1.0.0.0 
nmake install
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Xcode" -DMAGIC_XE_ROOT_DIR=/Users/ws/Downloads/MagicXE_Darwin_arm64_1.6.0.1
xcodebuild -project magic_xe_samples.xcodeproj -target install -configuration Release
export LD_LIBRARY_PATH+=:/trt_lib
cmake .. -DCMAKE_BUILD_TYPE=Release -DMAGIC_XE_ROOT_DIR=\Users\yanquan\Desktop\code\MagicXE_Linux_x86_1.0.0.0
make -j64
make install


四、Windows使用 微软Visual Studio IDE 自动安装.
cmd = r'""%s" INSTALL.vcxproj /ReBuild "%s|%s" /project PACKAGE /Out %s"' % (
        ide_path, build_config, build_arch, log_file)
print(cmd)
res = os.system(cmd)