#ifndef ENV_IMPLEMENT_H_
#define ENV_IMPLEMENT_H_

#include "magic_xe_env.h"
#include "magic_xe_env_internal.h"
#include "plugin/magic_xe_plugin.h"

namespace magic_xe {

class EnvImplementPrivate;
class EnvImplement {
public:
    EnvImplement(const MagicXEEnv *env);
    ~EnvImplement();

    bool IsInitSuccessed() const;
    MagicXEError EnvDeviceUpdate(const MagicXEEnvDevice *env_dev);
    const MagicXEEnv *EnvGet() const;
    const char *VersionGet() const;

    MagicXEError PluginLoad(const char *name, const MagicXEPlugin **plugin);
    const MagicXEPlugin *PluginLoad(MagicXEPluginType type) const;
    const MagicXEPlugin *PluginFind(MagicXEPluginType type) const;
    const MagicXEPlugin *PluginFind(const char *name) const;
    const MagicXEPlugin *PluginFind(MagicXEModelType model_type) const;
    const MagicXEPlugin *PluginFind(MagicXEDeviceType device_type) const;
    void PluginUnload(const MagicXEPlugin **plugin);

    MagicXEError ModelCreate(const char *utf8_model_file, MagicXEModel *handle);
    void ModelDestroy(MagicXEModel *handle);

private:
    EnvImplementPrivate *private_ = nullptr;
};

} // namespace magic_xe

#endif // ENV_IMPLEMENT_H_