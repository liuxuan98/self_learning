

typedef struct PipelineModelJson {
    const char *model_path;
    MagicXECustomRuntime runtime;
} PipelineModelJson;

typedef struct PipelineJsonObj {
    int plugin_reuse_idx; //only support -1

    MagicXEPluginType plugin_type;
    int num_delayer;
    int num_thread;
    int frame_from_idx;
    int frame_to_idx;
    int data_from_idx;

    bool output_frame;
    int output_pix_fmt;
    int output_memory;
    int output_width;
    int output_height;

    char *plugin_name;
    char *models_json;

    unsigned int arr_size;
    PipelineModelJson *arr;
} PipelineJsonObj;  //保存pipeline中的每个单模块算法的json信息



typedef struct PipelineTasks {
    unsigned int task_size;
    PipelineTask *task_arr;
    std::mutex *mutex;
    std::condition_variable *cond;
    MagicXEList *input_list;
    MagicXEList *output_list;
    volatile bool is_exited;
    MagicXEError mgxe_err;
} PipelineTasks;


typedef struct PipelinePlugin {
    unsigned int plugin_idx;
    PipelineJsonObj *json_obj;
    PipelineTasks *tasks;
    std::mutex *poll_mutex;
    std::condition_variable *poll_cond;
    std::condition_variable *output_cond;
    bool is_last_plugin;
} PipelinePlugin;