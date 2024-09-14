#include <mutex>
static magic_xe::RWLock g_env_lock;
static magic_xe::EnvImplement *g_env_implement = NULL;
MagicXEError MagicXEEnvInitialize(const MagicXEEnv *env){

    //key code.
    magic_xe::LockWriteGuard<magic_xe::RWLock> write_guard(g_env_lock); //LockWriteGuard自动管理锁的加锁和释放，高效灵活.
    static std::once_flag once;
    std::call_once(once,
        [](const MagicXEEnv *env) {
            g_env_implement = new magic_xe::EnvImplement(env);
            if (!g_env_implement->IsInitSuccessed()) {
                MAGIC_XE_LOGE("EnvImplement Init failed");
                delete g_env_implement;
                g_env_implement = NULL;
            }
        },
        env);

}