#include "env_implement.h"

#ifdef __APPLE__
#include <sys/syslimits.h>
#elif _WIN32
#include <Windows.h>
#endif

#include <string.h>

#include <list>
#include <map>
#include <string>




class EnvImplementPrivate {
public:
    EnvImplementPrivate(const MagicXEEnv *src_env) {

        sprintf(version_,
            "%d.%d.%d.%d",
            LIBMAGIC_XE_VERSION_MAJOR,
            LIBMAGIC_XE_VERSION_MINOR,
            LIBMAGIC_XE_VERSION_PATCH,
            LIBMAGIC_XE_VERSION_BUILD);

        env_ = (MagicXEEnv *)malloc(sizeof(MagicXEEnv));
        if (env_ == nullptr) {
            MAGIC_XE_LOGE("malloc MagicXEEnv failed");
            return;
        }
        memset(env_, 0, sizeof(MagicXEEnv));

        if (src_env->log_path != nullptr && strlen(src_env->log_path) > 0) {
            env_->log_path = (const char *)malloc(strlen(src_env->log_path) + 1);
            if (env_->log_path == nullptr) {
                MAGIC_XE_LOGE("malloc log_path:%s failed", src_env->log_path);
                return;
            }
            strcpy((char *)env_->log_path, src_env->log_path);
        }

        env_->license_file = (const char *)malloc(strlen(src_env->license_file) + 1);
        if (env_->license_file == nullptr) {
            MAGIC_XE_LOGE("malloc license_file:%s failed", src_env->license_file);
            return;
        }
        strcpy((char *)env_->license_file, src_env->license_file);

        env_->plugin_path = (const char *)malloc(strlen(src_env->plugin_path) + 1);
        if (env_->plugin_path == nullptr) {
            MAGIC_XE_LOGE("malloc plugin_path:%s failed", src_env->plugin_path);
            return;
        }
        strcpy((char *)env_->plugin_path, src_env->plugin_path);

        EnvDeviceUpdate(&src_env->env_dev);

        env_init_successed_ = true;
    }

    ~EnvImplementPrivate() {

        PluginCheck();

        if (env_ != nullptr) {
            if (env_->log_path != nullptr) {
                free((void *)env_->log_path);
                env_->log_path = nullptr;
            }
            if (env_->license_file != nullptr) {
                free((void *)env_->license_file);
                env_->license_file = nullptr;
            }
            if (env_->plugin_path != nullptr) {
                free((void *)env_->plugin_path);
                env_->plugin_path = nullptr;
            }
            free(env_);
            env_ = nullptr;
        }
    }

    void EnvDeviceUpdate(const MagicXEEnvDevice *env_dev) {

        env_->env_dev.opencl_device_id = env_dev->opencl_device_id;
        env_->env_dev.opencl_context = env_dev->opencl_context;

        env_->env_dev.opengl_gpu_type = env_dev->opengl_gpu_type;
        env_->env_dev.opengl_command_queue = env_dev->opengl_command_queue;

        env_->env_dev.cuda_device_id = env_dev->cuda_device_id;
        env_->env_dev.cuda_stream = env_dev->cuda_stream;
    }

    void PluginCheck() {

        LockReadGuard<RWLock> read_guard(plugin_lock_);
        std::map<std::string, PluginImplement *>::iterator iter = plugin_map_.begin();
        while (iter != plugin_map_.end()) {
            MAGIC_XE_LOGE("name:%s plugin:%p is not unload", iter->first.c_str(), iter->second->GetPlugin());
            iter++;
        }
    }

    MagicXEEnv *env_ = nullptr;
    bool env_init_successed_ = false;

    std::map<std::string, PluginImplement *> plugin_map_;
    RWLock plugin_lock_;

    char version_[17] = {0};
};

MagicXEError EnvImplement::PluginLoad(const char *name, const MagicXEPlugin **plugin) {

    LockWriteGuard<RWLock> write_guard(private_->plugin_lock_); //加写锁
    std::map<std::string, PluginImplement *>::iterator iter = private_->plugin_map_.begin();
    while (iter != private_->plugin_map_.end()) {
        if (0 == strcmp(iter->first.c_str(), name)) {
            *plugin = iter->second->GetPlugin();
            iter->second->Retain();
            return MAGIC_XE_SUCCESS;
        }
        iter++;
    }

    PluginImplement *plugin_imp = NULL;
    MagicXEError mgxe_err = PluginImplement::Create(private_->env_->plugin_path, name, &plugin_imp);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        return mgxe_err;
    }

    private_->plugin_map_[name] = plugin_imp; //插件加载到全局环境中,plugin_imp存储函数接口
    *plugin = plugin_imp->GetPlugin();
    return MAGIC_XE_SUCCESS;
}
