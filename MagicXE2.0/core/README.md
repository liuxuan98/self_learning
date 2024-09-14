# 是插件化框架的核心部分，会创建全局资源管理的调度类.

1.环境创建,静态单例类static magic_xe::RWLock g_env_lock; static magic_xe::EnvImplement *g_env_implement = NULL;

2.环境执行类,顶级管理类

3.插件执行类,次级管理类