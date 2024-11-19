## pipehandle(执行handle)

typedef struct MagicXEPipelineImp {
    // bool async;
    // unsigned int async_frame_max;
    unsigned int called_size;   //pipeline的调用数量
    unsigned int input_size;    //pipeline输入数量
    unsigned int output_size;   //pipeline输出数量
    unsigned int plugin_num;    //单点算法插件(handle) 数量
    PipelinePlugin *plugins;   //单点任务handle
    std::mutex *mutex;
    std::condition_variable *output_cond; //类似于生产者条件变量
    std::condition_variable *poll_cond;   //管理线程的条件变量
    std::future<void> *thread; //管理线程任务函数
    volatile bool is_exited;   //是否结束handle任务数量 ，理应input_size== output_size结束handle任务
} MagicXEPipelineImp;




//任务handle 异步线程函数PipelineTaskFunc，从输入列表中获取数据，如果没有则等待管理线程进行任务条件变量唤醒，有则进行处理并插入到输出链表的输出数据(用了链表排序)，插入和取数据都要加锁防止数据同步问题，

//管理hanle的异步线程函数MagicXEPipelinePollPlugins,来协调任务之间的数据流向并进行异步线程调度(改变状态并且数据装配好)

//主线程进行MagicXEPipelineProcess通过(MagicXEPipelinePluginsInputSet)一系列条件来设置插件任务状态(PIPELINE_INPUT_STATUS_WAITING等),并且装配数据,来唤醒插件任务进行处理;主线程MagicXEPipelineOutputProcess函数来装配输出


https://www.parallellabs.com/ 学习博主私人链接
