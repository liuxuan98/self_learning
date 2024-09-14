
#ifdef _WIN32  //dll 库加载头文件
#define NOMINMAX
#include <libloaderapi.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif


static std::string GetDLLError() {
    LPVOID lpMsgBuf = NULL;
    DWORD err = GetLastError(); //加载dll 错误获取api

    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&lpMsgBuf,
        0,
        NULL);
    std::string ret((char *)lpMsgBuf);
    return ret;
}
#else
static std::string GetDLLError() {
    char *str = dlerror();
    std::string ret(str);
    return ret;
}
#endif

namespace magic_xe {
class PluginImplementPrivate {
public:
    PluginImplementPrivate(const char *plugin_path, const char *name) {

        memset(&plugin_, 0, sizeof(MagicXEPlugin));
#ifdef MAGIC_XE_ENABLE_SHARED_LIB
        if (!LoadDllLibrary(plugin_path, name)) {
            MAGIC_XE_LOGE("LoadDllLibrary failed!");
            load_err_ = MAGIC_XE_PLUGIN_FAIL_LOAD;
            return;
        }
#endif
        MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
#define LOAD_PLUGIN_NAME(type_name)                                       \
    mgxe_err = LoadPlugin(name, type_name, &plugin_);                     \
    if (mgxe_err != MAGIC_XE_SUCCESS && mgxe_err != MAGIC_XE_NOT_FOUND) { \
        MAGIC_XE_LOGE("LoadPlugin %s failed:%d", type_name, mgxe_err);    \
        load_err_ = mgxe_err;                                             \
        return;                                                           \
    }
        LOAD_PLUGIN_NAME("decoder");
        LOAD_PLUGIN_NAME("encoder");
        LOAD_PLUGIN_NAME("neural_net");
        LOAD_PLUGIN_NAME("device");
        LOAD_PLUGIN_NAME("filter");
        LOAD_PLUGIN_NAME("adapter");
        LOAD_PLUGIN_NAME("resolver");
        LOAD_PLUGIN_NAME("trans");
        LOAD_PLUGIN_NAME("delayer");

        if (plugin_.types == 0) {
            MAGIC_XE_LOGE("LoadPlugin types is empty!");
            load_err_ = MAGIC_XE_NOT_IMPLEMENTED;
        }
    }

    ~PluginImplementPrivate() {
#ifdef MAGIC_XE_ENABLE_SHARED_LIB
        UnloadDllLibrary();
#endif
        MAGIC_XE_FREE(plugin_.decoder);
        MAGIC_XE_FREE(plugin_.encoder);
        MAGIC_XE_FREE(plugin_.neural_net);
        MAGIC_XE_FREE(plugin_.device);
        MAGIC_XE_FREE(plugin_.filter);
        MAGIC_XE_FREE(plugin_.adapter);
        MAGIC_XE_FREE(plugin_.resolver);
        MAGIC_XE_FREE(plugin_.trans);
        MAGIC_XE_FREE(plugin_.delayer);
    }

#ifdef MAGIC_XE_ENABLE_SHARED_LIB
    bool LoadDllLibrary(const char *plugin_path, const char *name);
    void UnloadDllLibrary();
#endif
    MagicXEError LoadPlugin(const char *name, const char *type_name, MagicXEPlugin *plugin);

    MagicXEError load_err_ = MAGIC_XE_SUCCESS;
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void *handle_ = nullptr;
#endif
    volatile int count_ = 1; //volatile变量是一种特殊类型的变量，告诉编译器不要优化这个变量，保证内存的可见性.
    MagicXEPlugin plugin_;
};


MagicXEError PluginImplementPrivate::LoadPlugin(const char *name, const char *type_name, MagicXEPlugin *plugin) {

    std::string func_name = std::string(name) + "_" + type_name;
    MagicXEPluginInitFunc func = nullptr;

#ifdef MAGIC_XE_ENABLE_SHARED_LIB
#ifdef _WIN32
    func = (MagicXEPluginInitFunc)GetProcAddress(handle_, func_name.c_str());
#else
    func = (MagicXEPluginInitFunc)dlsym(handle_, func_name.c_str());
#endif
#else
    func = MagicXEStaticPluginInitFuncGet(func_name.c_str());
#endif
    if (func == nullptr) {
        //MAGIC_XE_LOGE("load symbol:%s failed:%s!", func_name.c_str(), GetDLLError().c_str());
        return MAGIC_XE_NOT_FOUND;
    }

    MagicXEError mgxe_err = func(name, plugin);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("run function:%s failed:%d!", func_name.c_str(), mgxe_err);
        return mgxe_err;
    }

    return mgxe_err;
}


MagicXEError PluginImplement::Create(const char *plugin_path, const char *name, PluginImplement **plugin_imp) {

    MAGIC_XE_LOGI("create plugin:%s", name);

    *plugin_imp = new PluginImplement(plugin_path, name);
    MagicXEError mgxe_err = (*plugin_imp)->LoadStatus();
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("plugin_path:%s name:%s load failed:%d", plugin_path, name, mgxe_err);
        delete *plugin_imp;
        *plugin_imp = nullptr;
        return mgxe_err;
    }

    return MAGIC_XE_SUCCESS;
}


PluginImplement::PluginImplement(const char *plugin_path, const char *name)
    : private_(new PluginImplementPrivate(plugin_path, name)) {
}


bool PluginImplementPrivate::LoadDllLibrary(const char *plugin_path, const char *name) {

    std::string full_lib_name(plugin_path);
#ifdef _WIN32
    full_lib_name += "/" + std::string(name) + ".dll";
#elif __APPLE__
    full_lib_name += "/lib" + std::string(name) + ".dylib";
#elif __linux__
    full_lib_name += "/lib" + std::string(name) + ".so";
#endif

#ifdef _WIN32
    wchar_t dll_path[MAX_PATH + 1] = {0};
    GetDllDirectoryW(MAX_PATH, dll_path);
    std::wstring w_lib_dir_path = MultiByteCharToWString(plugin_path);
    SetDllDirectoryW(w_lib_dir_path.c_str());
    std::wstring w_lib_full_path = MultiByteCharToWString(full_lib_name.c_str());
    handle_ = LoadLibraryW(w_lib_full_path.c_str());
#else
    handle_ = dlopen(full_lib_name.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif

    if (handle_ == nullptr) {
        MAGIC_XE_LOGE("load library:%s failed:%s!", full_lib_name.c_str(), GetDLLError().c_str());
    }
#ifdef _WIN32
    SetDllDirectoryW(dll_path);
#endif
    return handle_ != nullptr;
}

void PluginImplementPrivate::UnloadDllLibrary() {

    if (handle_ != nullptr) {
#ifdef _WIN32
        if (!FreeLibrary(handle_)) {
#else
        if (dlclose(handle_) != 0) {
#endif
            MAGIC_XE_LOGE("unload library:%s failed!", plugin_.name);
        }
        handle_ = nullptr;
    }
}









}