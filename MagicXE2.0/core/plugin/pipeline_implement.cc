

MagicXEError PipelinePluginTasksCreate(PipelinePlugin *plugin, PipelinePlugin *reuse_plugin) {

    if (plugin == nullptr || plugin->json_obj == nullptr) {
        MAGIC_XE_LOGE("plugin:%p or json_obj:%p is nullptr", plugin, plugin->json_obj);
        return MAGIC_XE_INVALID_PIPELINE;
    }
    MAGIC_XE_LOGI(PLUGIN_LOG_TAG "reuse_plugin:%p Enter", PLUGIN_LOG_ARG, reuse_plugin);

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    PipelineJsonObj *json_obj = plugin->json_obj;
    unsigned int task_size = json_obj->num_thread;
    plugin->tasks = (PipelineTasks *)malloc(sizeof(PipelineTasks));
    if (plugin->tasks == nullptr) {
        MAGIC_XE_LOGE(PLUGIN_LOG_TAG "malloc sizeof(PipelineTasks) failed!", PLUGIN_LOG_ARG);
        return MAGIC_XE_OUT_OF_MEMORY;
    }
    memset(plugin->tasks, 0, sizeof(PipelineTasks));

    plugin->tasks->task_arr = (PipelineTask *)malloc(task_size * sizeof(PipelineTask));
    if (plugin->tasks->task_arr == nullptr) {
        MAGIC_XE_LOGE("malloc task_size:%d * sizeof(PipelineTask) failed!", task_size);
        mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
        goto __end;
    }
    memset(plugin->tasks->task_arr, 0, task_size * sizeof(PipelineTask));
    plugin->tasks->task_size = task_size;
    plugin->tasks->is_exited = false;
    plugin->tasks->mutex = new std::mutex;
    plugin->tasks->cond = new std::condition_variable;

    if ((mgxe_err = PipelinePluginTasksOpen(plugin, reuse_plugin)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE(PLUGIN_LOG_TAG "PipelinePluginTasksOpen failed:%d", PLUGIN_LOG_ARG, mgxe_err);
        goto __end;
    }
    if ((mgxe_err = MagicXEListCreate(&plugin->tasks->input_list)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE(PLUGIN_LOG_TAG "MagicXEMagicXEListCreate failed:%d", PLUGIN_LOG_ARG, mgxe_err);
        goto __end;
    }
    if ((mgxe_err = MagicXEListCreate(&plugin->tasks->output_list)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE(PLUGIN_LOG_TAG "MagicXEMagicXEListCreate failed:%d", PLUGIN_LOG_ARG, mgxe_err);
        goto __end;
    }

    for (int i = 0; i < task_size; ++i) {
        PipelineTask *task = &plugin->tasks->task_arr[i];
        task->thread = new std::future<void>;
        *task->thread = std::async(std::launch::async, PipelineTaskFunc, plugin, i);
    }

__end:
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        PipelinePluginTasksDestroy(plugin);
        return mgxe_err;
    }

    MAGIC_XE_LOGI(PLUGIN_LOG_TAG "reuse_plugin:%p Leave", PLUGIN_LOG_ARG, reuse_plugin);
    return MAGIC_XE_SUCCESS;
}