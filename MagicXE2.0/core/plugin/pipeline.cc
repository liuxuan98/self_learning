#include "pipeline.h"


//关键的代码
typedef struct MagicXEPipelineImp {
    bool async;
    unsigned int async_frame_max;
    unsigned int called_size;
    unsigned int input_size;
    unsigned int output_size;
    unsigned int plugin_num;
    PipelinePlugin *plugins;
    char *config_schemas;
    char *config_values;
    std::mutex *mutex;
    std::condition_variable *output_cond;
    std::condition_variable *poll_cond;
    std::future<void> *thread;
    volatile bool is_exited;
} MagicXEPipelineImp;

static MagicXEError MagicXEPipelinePluginParserAttribute(
    MagicXEJsonObject models_obj, PipelineJsonObj *json_obj, PipelinePlugin *plugin, int idx) {

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    int plugin_reuse_idx = -1;
    unsigned int model_size = 0;
    MagicXEPluginType plugin_type = MAGIC_XE_PLUGIN_TYPE_NONE;
    const char *plugin_name = nullptr;
    char *models_json = nullptr;
    unsigned int task_size = 1;
    int num_delayer = 0;
    int num_thread = 1;
    bool output_frame = true;
    int output_pix_fmt = PIPELINE_FRAME_OUTPUT_EQ_EI;
    int output_memory = PIPELINE_FRAME_OUTPUT_EQ_EI;
    int output_width = PIPELINE_FRAME_OUTPUT_EQ_EI;
    int output_height = PIPELINE_FRAME_OUTPUT_EQ_EI;
    MagicXEJsonArray model_arr = nullptr;

    plugin_type = (MagicXEPluginType)MagicXEJsonIntGet(
        MagicXEJsonObjectGet(models_obj, "plugin_type"), MAGIC_XE_PLUGIN_TYPE_NONE);
    if (plugin_type == MAGIC_XE_PLUGIN_TYPE_NONE || !ValidateModelPluginType(plugin_type)) {
        MAGIC_XE_LOGE("models_obj:%p error plugin type:0X%X", models_obj, plugin_type);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }

    plugin_name = MagicXEJsonStringGet(MagicXEJsonObjectGet(models_obj, "plugin_name"));
    if (plugin_name == nullptr || strlen(plugin_name) <= 0) {
        MAGIC_XE_LOGE("models_obj:%p no model array", models_obj);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }

    models_json = MagicXEJsonObjectToStringDup(models_obj);
    if (models_json == nullptr || strlen(models_json) <= 0) {
        MAGIC_XE_LOGE("plugin_name:%s json is error", plugin_name);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }

    if (plugin_type == MAGIC_XE_PLUGIN_TYPE_DELAYER) {
        num_delayer = MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "num_delayer"), 1);
        if (num_delayer <= 0) {
            MAGIC_XE_LOGE("plugin_name:%s num_delayer:%d json is error, delayer must > 0", plugin_name, num_delayer);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
    }

    num_thread = MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "num_thread"), 1);
    if (num_thread <= 0 || num_thread > 16) {
        MAGIC_XE_LOGE("plugin_name:%s num_thread:%d json is error", plugin_name, num_thread);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }
    if (plugin_type == MAGIC_XE_PLUGIN_TYPE_DELAYER && num_thread != 1) {
        MAGIC_XE_LOGE(
            "plugin_name:%s num_thread:%d json is error, delayer only support one thread", plugin_name, num_thread);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }

    output_frame = MagicXEJsonBoolGet(MagicXEJsonObjectGet(models_obj, "output_frame"), true);
    if (output_frame) {
        output_pix_fmt =
            MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "output_pix_fmt"), PIPELINE_FRAME_OUTPUT_EQ_EI);
        if (output_pix_fmt != PIPELINE_FRAME_OUTPUT_EQ_EI && output_pix_fmt != PIPELINE_FRAME_OUTPUT_EQ_PI
            && !ValidateFramePixelFormat((MagicXEPixelFormat)output_pix_fmt)) {
            MAGIC_XE_LOGE("plugin_name:%s output_pix_fmt:%d idx:%d json is error", plugin_name, output_pix_fmt, idx);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
        output_memory =
            MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "output_memory"), PIPELINE_FRAME_OUTPUT_EQ_EI);
        if (output_memory != PIPELINE_FRAME_OUTPUT_EQ_EI && output_memory != PIPELINE_FRAME_OUTPUT_EQ_PI
            && !ValidateFrameMemoryType((MagicXEMemoryType)output_memory)) {
            MAGIC_XE_LOGE("plugin_name:%s output_memory:%d json is error", plugin_name, output_memory);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
        output_width = MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "output_width"), 0);
        output_height = MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "output_height"), 0);
        if ((output_width <= 0 && output_width != PIPELINE_FRAME_OUTPUT_EQ_EI
                && output_width != PIPELINE_FRAME_OUTPUT_EQ_PI)
            || (output_height <= 0 && output_height != PIPELINE_FRAME_OUTPUT_EQ_EI
                   && output_height != PIPELINE_FRAME_OUTPUT_EQ_PI)) {
            MAGIC_XE_LOGE("plugin_name:%s output_width:%d output_height:%d json is error",
                plugin_name,
                output_width,
                output_height);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
    }

    mgxe_err =
        MagicXEPipelinePluginParserNecessary(models_obj, json_obj, plugin, idx, plugin_name, plugin_type, output_frame);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPipelinePluginParserNecessary idx:%d failed:%d", idx, mgxe_err);
        goto __end;
    }

    json_obj->plugin_type = plugin_type;
    json_obj->num_delayer = num_delayer;
    json_obj->num_thread = num_thread;

    json_obj->output_frame = output_frame;
    json_obj->output_pix_fmt = output_pix_fmt;
    json_obj->output_memory = output_memory;
    json_obj->output_width = output_width;
    json_obj->output_height = output_height;

    json_obj->plugin_name = strdup(plugin_name);
    json_obj->models_json = strdup(models_json);

    model_arr = MagicXEJsonArrayAlloc(MagicXEJsonObjectGet(models_obj, "models"));
    model_size = MagicXEJsonArraySize(model_arr);
    if (model_arr == nullptr || model_size <= 0) {
        //No Need Models
        goto __end;
    }

    json_obj->arr = (PipelineModelJson *)malloc(model_size * sizeof(PipelineModelJson));
    if (json_obj->arr == nullptr) {
        MAGIC_XE_LOGE("malloc model_size:%d PipelineModelJson failed!", model_size);
        mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
        goto __end;
    }
    json_obj->arr_size = model_size;
    memset(json_obj->arr, 0, model_size * sizeof(PipelineModelJson));

    for (unsigned int i = 0; i < model_size; i++) {
        MagicXEJsonObject model_obj = MagicXEJsonArrayAt(model_arr, i);
        MagicXEJsonObject path_obj = MagicXEJsonObjectGet(model_obj, "model_path");
        const char *model_path = MagicXEJsonStringGet(path_obj);
        if (model_path == nullptr || strlen(model_path) <= 0) {
            MAGIC_XE_LOGE("models_obj:%p Get model path failed", models_obj);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
        json_obj->arr[i].model_path = strdup(model_path);
        if (json_obj->arr[i].model_path == nullptr) {
            MAGIC_XE_LOGE("malloc model_path:%s failed!", model_path);
            mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
            goto __end;
        }
        json_obj->arr[i].runtime.num_thread = MagicXEJsonIntGet(MagicXEJsonObjectGet(model_obj, "num_thread"), 0);
        json_obj->arr[i].runtime.device_type = (MagicXEDeviceType)MagicXEJsonIntGet(
            MagicXEJsonObjectGet(model_obj, "device_type"), MAGIC_XE_DEVICE_TYPE_NONE);
    }

__end:
    MAGIC_XE_FREE(models_json);
    MagicXEJsonArrayFree(model_arr);
    return mgxe_err;
}


static MagicXEError MagicXEPipelineModelsParser(MagicXEJsonObject models_obj, PipelinePlugin *plugin, int idx) {

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    int plugin_reuse_idx = -1;
    PipelineJsonObj **arr = &plugin[idx].json_obj;

    MAGIC_XE_LOGD("models_obj:%p idx:%d Enter", models_obj, idx);
#if 0
    plugin_reuse_idx = (MagicXEPluginType)MagicXEJsonIntGet(MagicXEJsonObjectGet(models_obj, "plugin_reuse_idx"), -1);
    if (plugin_reuse_idx >= 0 && plugin_reuse_idx >= idx) {
        MAGIC_XE_LOGE("models_obj:%p plugin_reuse_idx:%d idx:%d json is error", models_obj, plugin_reuse_idx, idx);
        return MAGIC_XE_INVALID_JSON;
    }
    if (plugin_reuse_idx >= 0 && plugin[plugin_reuse_idx].json_obj->plugin_reuse_idx >= 0) {
        MAGIC_XE_LOGE("models_obj:%p plugin_reuse_idx:%d idx:%d plugin reuse idex:%d != -1",
            models_obj,
            plugin_reuse_idx,
            idx,
            plugin[plugin_reuse_idx].json_obj->plugin_reuse_idx);
        return MAGIC_XE_INVALID_JSON;
    }
#endif
    *arr = (PipelineJsonObj *)malloc(sizeof(PipelineJsonObj));
    if (*arr == nullptr) {
        MAGIC_XE_LOGE("malloc MagicXEModelJsonArray failed!");
        mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
        goto __end;
    }
    memset(*arr, 0, sizeof(PipelineJsonObj));

    (*arr)->plugin_reuse_idx = plugin_reuse_idx;
    if (plugin_reuse_idx < 0) {
        mgxe_err = MagicXEPipelinePluginParserAttribute(models_obj, *arr, plugin, idx);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MagicXEPipelinePluginParser idx:%d failed:%d", idx, mgxe_err);
            goto __end;
        }
    } else {
        PipelineJsonObj *reuse_obj = plugin[plugin_reuse_idx].json_obj;
        if (reuse_obj->num_thread > 1) {
            MAGIC_XE_LOGE("plugin:%s plugin_reuse_idx:%d idx:%d json is error, num_thread:%d must be 1",
                reuse_obj->plugin_name,
                plugin_reuse_idx,
                idx,
                reuse_obj->num_thread);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }

        mgxe_err = MagicXEPipelinePluginParserNecessary(
            models_obj, *arr, plugin, idx, reuse_obj->plugin_name, reuse_obj->plugin_type, reuse_obj->output_frame);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MagicXEPipelinePluginParserNecessary idx:%d failed:%d", idx, mgxe_err);
            goto __end;
        }

        (*arr)->plugin_reuse_idx = plugin_reuse_idx;
        (*arr)->plugin_type = reuse_obj->plugin_type;
        (*arr)->num_delayer = reuse_obj->num_delayer;
        (*arr)->num_thread = reuse_obj->num_thread;
        (*arr)->frame_from_idx = reuse_obj->frame_from_idx;
        (*arr)->frame_to_idx = reuse_obj->frame_to_idx;
        (*arr)->data_from_idx = reuse_obj->data_from_idx;
        (*arr)->output_frame = reuse_obj->output_frame;
        (*arr)->output_pix_fmt = reuse_obj->output_pix_fmt;
        (*arr)->output_memory = reuse_obj->output_memory;
        (*arr)->output_width = reuse_obj->output_width;
        (*arr)->output_height = reuse_obj->output_height;
        (*arr)->plugin_name = strdup(reuse_obj->plugin_name);
        (*arr)->models_json = strdup(reuse_obj->models_json);
        if ((*arr)->plugin_name == nullptr || (*arr)->models_json == nullptr) {
            MAGIC_XE_LOGE("plugin_name:%s dst:%p models_json:%s dst:%p strdup failed!",
                reuse_obj->plugin_name,
                (*arr)->plugin_name,
                reuse_obj->models_json,
                (*arr)->models_json);
            mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
            goto __end;
        }

        if (reuse_obj->arr_size > 0) {
            (*arr)->arr = (PipelineModelJson *)malloc(reuse_obj->arr_size * sizeof(PipelineModelJson));
            if ((*arr)->arr == nullptr) {
                MAGIC_XE_LOGE("malloc model_size:%d PipelineModelJson failed!", reuse_obj->arr_size);
                mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
                goto __end;
            }
            (*arr)->arr_size = reuse_obj->arr_size;
            memset((*arr)->arr, 0, reuse_obj->arr_size * sizeof(PipelineModelJson));
            for (unsigned int i = 0; i < reuse_obj->arr_size; i++) {
                (*arr)->arr[i].model_path = strdup(reuse_obj->arr[i].model_path);
                if ((*arr)->arr[i].model_path == nullptr) {
                    MAGIC_XE_LOGE("malloc model_path:%s failed!", reuse_obj->arr[i].model_path);
                    mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
                    goto __end;
                }
                (*arr)->arr[i].runtime.num_thread = reuse_obj->arr[i].runtime.num_thread;
                (*arr)->arr[i].runtime.device_type = reuse_obj->arr[i].runtime.device_type;
            }
        }
    }

__end:
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MagicXEPipelineModelsFree(arr);
    }
    MAGIC_XE_LOGD("models_obj:%p idx:%d mgxe_err:%d Leave", models_obj, idx, mgxe_err);
    return mgxe_err;
}


static MagicXEError MagicXEPipelineParser(const char *pipeline_json, MagicXEPipelineImp *pp_imp) {

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    MagicXEJsonHandle json_handle = nullptr;
    MagicXEJsonObject root_obj = nullptr;
    MagicXEJsonArray pipeline_arr = nullptr;
    unsigned int pipeline_size = 0;

    if (pipeline_json == nullptr || strlen(pipeline_json) <= 0 || pp_imp == nullptr) {
        MAGIC_XE_LOGE("pipeline_json:%p pp_imp:%p has nullptr", pipeline_json, pp_imp);
        return MAGIC_XE_INVALID_PARAM;
    }

    pp_imp->plugins = nullptr;
    if ((mgxe_err = MagicXEJsonCreate(nullptr, (void *)pipeline_json, strlen(pipeline_json), &json_handle))
        != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("pipeline_json:%s MagicXEJsonCreate failed:%d", pipeline_json, mgxe_err);
        goto __end;
    }

    root_obj = MagicXEJsonRootGet(json_handle);

    pp_imp->async = MagicXEJsonBoolGet(MagicXEJsonObjectGet(root_obj, "async"), false);
    if (pp_imp->async) {
        pp_imp->async_frame_max = MagicXEJsonIntGet(MagicXEJsonObjectGet(root_obj, "async_frame_max"), 1);
        if (pp_imp->async_frame_max <= 0 || pp_imp->async_frame_max > 100) {
            MAGIC_XE_LOGE("pipeline is async, but async_frame_max:%d json is error", pp_imp->async_frame_max);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            goto __end;
        }
    }

    pipeline_arr = MagicXEJsonArrayAlloc(MagicXEJsonObjectGet(root_obj, "pipeline"));
    pipeline_size = MagicXEJsonArraySize(pipeline_arr);
    if (pipeline_arr == nullptr || pipeline_size <= 1) {
        MAGIC_XE_LOGE("pipeline_json:%s no model array or pipeline_size:%d <= 1", pipeline_json, pipeline_size);
        mgxe_err = MAGIC_XE_INVALID_JSON;
        goto __end;
    }

    pp_imp->plugins = (PipelinePlugin *)malloc(pipeline_size * sizeof(PipelinePlugin));
    if (pp_imp->plugins == nullptr) {
        MAGIC_XE_LOGE("malloc pipeline_size:%u PipelinePlugin failed!", pipeline_size);
        mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
        goto __end;
    }
    pp_imp->plugin_num = pipeline_size;
    memset(pp_imp->plugins, 0, pipeline_size * sizeof(PipelinePlugin));

    for (int i = 0; i < pipeline_size; ++i) {
        MagicXEJsonObject ai_obj = MagicXEJsonArrayAt(pipeline_arr, i);
        if (ai_obj == nullptr) {
            MAGIC_XE_LOGE("MagicXEJsonArrayAt i:%d return empty!", i);
            mgxe_err = MAGIC_XE_INVALID_JSON;
            break;
        }
        if ((mgxe_err = MagicXEPipelineModelsParser(ai_obj, pp_imp->plugins, i)) != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MagicXEPipelineModelsParser i:%d failed:%d", i, mgxe_err);
            break;
        }
    }

__end:
    MagicXEJsonArrayFree(pipeline_arr);
    MagicXEJsonDestory(&json_handle);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        for (int i = 0; pp_imp->plugins != nullptr && i < pp_imp->plugin_num; i++) {
            MagicXEPipelineModelsFree(&pp_imp->plugins[i].json_obj);
        }
        MAGIC_XE_FREE(pp_imp->plugins);
    }
    return mgxe_err;
}

MagicXEError MagicXEPipelineOpen(const char *pipeline_json, MagicXEPipeline *pipeline){
     if (pipeline_json == nullptr || strlen(pipeline_json) <= 0 || pipeline == nullptr) {
        MAGIC_XE_LOGE("pipeline_json:%p pipeline:%p has nullptr", pipeline_json, pipeline);
        return MAGIC_XE_INVALID_PARAM;
    }

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    MagicXEPipelineImp *pipeline_imp = (MagicXEPipelineImp *)malloc(sizeof(MagicXEPipelineImp));
    if (pipeline_imp == nullptr) {
        MAGIC_XE_LOGE("malloc MagicXEPipelineImp failed");
        mgxe_err = MAGIC_XE_OUT_OF_MEMORY;
        goto __end;
    }
    memset(pipeline_imp, 0, sizeof(MagicXEPipelineImp));
    *pipeline = pipeline_imp;

    if ((mgxe_err = MagicXEPipelineParser(pipeline_json, pipeline_imp)) != MAGIC_XE_SUCCESS) { //MagicXEPipelineParser的json解析
        MAGIC_XE_LOGE("MagicXEPipelineParser failed:%d", mgxe_err);
        goto __end;
    }
    //pipeline 管理线程的
    pipeline_imp->is_exited = false;
    pipeline_imp->mutex = new std::mutex; 
    pipeline_imp->output_cond = new std::condition_variable;
    pipeline_imp->poll_cond = new std::condition_variable;

    for (unsigned int i = 0; i < pipeline_imp->plugin_num; ++i) {
        PipelinePlugin *plugin = &pipeline_imp->plugins[i], *reuse_plugin = nullptr;
        plugin->plugin_idx = i;
        plugin->poll_mutex = pipeline_imp->mutex;   //插件1
        plugin->poll_cond = pipeline_imp->poll_cond;
        plugin->output_cond = pipeline_imp->output_cond;
        plugin->is_last_plugin = i == pipeline_imp->plugin_num - 1;

        PipelineJsonObj *json_obj = plugin->json_obj;
        if (json_obj->plugin_reuse_idx >= 0) {
            reuse_plugin = &pipeline_imp->plugins[json_obj->plugin_reuse_idx];
        }
        if (!pipeline_imp->async && plugin->json_obj->num_thread > 1) {
            MAGIC_XE_LOGW(PLUGIN_LOG_TAG "num_thread:%d sync not support, change to default 1",
                PLUGIN_LOG_ARG,
                json_obj->num_thread);
            plugin->json_obj->num_thread = 1;
        }
        if ((mgxe_err = PipelinePluginTasksCreate(plugin, reuse_plugin)) != MAGIC_XE_SUCCESS) {  //创建关键任务
            MAGIC_XE_LOGE("plugin i:%u PipelinePluginTasksCreate failed:%d", i, mgxe_err);
            goto __end;
        }
    }

    pipeline_imp->thread = new std::future<void>; //管理线程调度 两个任务线程
    *pipeline_imp->thread = std::async(std::launch::async, MagicXEPipelinePollPlugins, pipeline_imp);

__end:
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MagicXEPipelineClose(pipeline);
    }
    return mgxe_err;

}
//管理线程
static void MagicXEPipelinePollPlugins(MagicXEPipelineImp *imp) {

    MAGIC_XE_LOGI("MagicXEPipelinePollPlugins:%p Enter", imp);
    unsigned long long poll_times = 0;
    while (!imp->is_exited) {
        /*Drive updates for each layer, no need for the first layer*/
        std::unique_lock<std::mutex> lg(*imp->mutex);
        MAGIC_XE_LOGD("MagicXEPipelinePollPlugins:%p Poll times:%llu Begin", imp, ++poll_times);
        for (unsigned int i = 1; i < imp->plugin_num; i++) {
            PipelinePlugin *plugin = &imp->plugins[i];
            for (unsigned int j = 0; j < imp->input_size; j++) {
                MagicXEPipelinePollPluginByOutput(imp, i, j);
            }
        }
        MAGIC_XE_LOGD("MagicXEPipelinePollPlugins:%p Poll times:%llu Leave", imp, poll_times);
        imp->poll_cond->wait(lg);
    }
    MAGIC_XE_LOGI("MagicXEPipelinePollPlugins:%p Leave", imp);
}


