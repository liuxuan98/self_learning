一、【demo】
    ''' 
     ncnn::Net mobilenet;

    mobilenet.opt.use_vulkan_compute = true;

    // model is converted from https://github.com/chuanqi305/MobileNet-SSD
    // and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    if (mobilenet.load_param("E:/NCNN/ncnn/protobuf_build/examples/Release/mobilenet_ssd_voc_ncnn.param"))//liuxuan add 绝对路径地址
        exit(-1);
    if (mobilenet.load_model("E:/NCNN/ncnn/protobuf_build/examples/Release/mobilenet_ssd_voc_ncnn.bin"))
        exit(-1);

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);
    //网络推理部分
    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);


    '''
1.网络加载
1.1 load_param 参数加载
1.2 load_model 模型加载
2.数据预处理
略
3.网络推理
3.1 create_extractor
3.2 input 输入blob进入
3.3 extract 输出blob


