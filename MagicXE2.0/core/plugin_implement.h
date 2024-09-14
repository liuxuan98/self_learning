#ifndef PLUGIN_IMPLEMENT_H_
#define PLUGIN_IMPLEMENT_H_

#include "plugin/magic_xe_plugin.h"

namespace magic_xe {

class PluginImplementPrivate;
class PluginImplement {
public:
    static MagicXEError Create(const char *plugin_path, const char *name, PluginImplement **plugin_imp);
    static MagicXEError Destroy(PluginImplement *plugin_imp);

    const MagicXEPlugin *GetPlugin() const;

    void Retain();
    int RetainCount();
    void Release();

protected:
    PluginImplement(const char *plugin_path, const char *name);
    ~PluginImplement();

    MagicXEError LoadStatus();

private:
    PluginImplementPrivate *private_ = nullptr;
};

} // namespace magic_xe

#endif // PLUGIN_IMPLEMENT_H_