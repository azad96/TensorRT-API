#include "resnet34.h"
#include <cmath>


bool loadWeights(const std::string& file, std::map<std::string, nvinfer1::Weights>& weightMap){
    std::cout << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file);
    if(!input.is_open()){
        std::cerr << "Unable to load weight file." << std::endl;
        return false;
    }

    // Read number of weight blobs
    int32_t count;
    input >> count;
    if(count <= 0){
        std::cerr << "Invalid weight std::map file." << std::endl;
        return false;
    }

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }
    return true;   
}


nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


nvinfer1::ITensor* basicBlock(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor* input, int inch, int outch, nvinfer1::DimsHW stride, std::string lname) {
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolution(*input, outch, nvinfer1::DimsHW{3, 3}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(stride);
    conv1->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    nvinfer1::IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, nvinfer1::DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    nvinfer1::IElementWiseLayer* ew1;
    if (inch != outch) {
        nvinfer1::IConvolutionLayer* conv3 = network->addConvolution(*input, outch, nvinfer1::DimsHW{1, 1}, weightMap[lname + ".downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStride(stride);
        nvinfer1::IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(*input, *bn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }

    nvinfer1::IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);
    return relu2->getOutput(0);
}


nvinfer1::ITensor* resnet34(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor* input, int outputCount){
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolution(*input, 64, nvinfer1::DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(nvinfer1::DimsHW{2, 2});
    conv1->setPadding(nvinfer1::DimsHW{3, 3});
    
    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    nvinfer1::IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(nvinfer1::DimsHW{2, 2});
    pool1->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::ITensor* bb1  = basicBlock(network, weightMap, pool1->getOutput(0), 64, 64, nvinfer1::DimsHW{1, 1}, "layer1.0");
    nvinfer1::ITensor* bb2  = basicBlock(network, weightMap, bb1,  64,  64,  nvinfer1::DimsHW{1, 1}, "layer1.1");
    nvinfer1::ITensor* bb3  = basicBlock(network, weightMap, bb2,  64,  64,  nvinfer1::DimsHW{1, 1}, "layer1.2");

    nvinfer1::ITensor* bb4  = basicBlock(network, weightMap, bb3,  64,  128, nvinfer1::DimsHW{2, 2}, "layer2.0");
    nvinfer1::ITensor* bb5  = basicBlock(network, weightMap, bb4,  128, 128, nvinfer1::DimsHW{1, 1}, "layer2.1");
    nvinfer1::ITensor* bb6  = basicBlock(network, weightMap, bb5,  128, 128, nvinfer1::DimsHW{1, 1}, "layer2.2");
    nvinfer1::ITensor* bb7  = basicBlock(network, weightMap, bb6,  128, 128, nvinfer1::DimsHW{1, 1}, "layer2.3");

    nvinfer1::ITensor* bb8  = basicBlock(network, weightMap, bb7,  128, 256, nvinfer1::DimsHW{2, 2}, "layer3.0");
    nvinfer1::ITensor* bb9  = basicBlock(network, weightMap, bb8,  256, 256, nvinfer1::DimsHW{1, 1}, "layer3.1");
    nvinfer1::ITensor* bb10 = basicBlock(network, weightMap, bb9,  256, 256, nvinfer1::DimsHW{1, 1}, "layer3.2");
    nvinfer1::ITensor* bb11 = basicBlock(network, weightMap, bb10, 256, 256, nvinfer1::DimsHW{1, 1}, "layer3.3");
    nvinfer1::ITensor* bb12 = basicBlock(network, weightMap, bb11, 256, 256, nvinfer1::DimsHW{1, 1}, "layer3.4");
    nvinfer1::ITensor* bb13 = basicBlock(network, weightMap, bb12, 256, 256, nvinfer1::DimsHW{1, 1}, "layer3.5");

    nvinfer1::ITensor* bb14 = basicBlock(network, weightMap, bb13, 256, 512, nvinfer1::DimsHW{2, 2}, "layer4.0");
    nvinfer1::ITensor* bb15 = basicBlock(network, weightMap, bb14, 512, 512, nvinfer1::DimsHW{1, 1}, "layer4.1");
    nvinfer1::ITensor* bb16 = basicBlock(network, weightMap, bb15, 512, 512, nvinfer1::DimsHW{1, 1}, "layer4.2");

    nvinfer1::IPoolingLayer* pool2 = network->addPooling(*bb16, nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{7, 7});
    assert(pool2);
    pool2->setStride(nvinfer1::DimsHW{1, 1});
    
    nvinfer1::IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), outputCount, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    return fc1->getOutput(0);
}


Resnet34::Resnet34(std::string &weight_path, int in_channel, int in_height, int in_width, int output_count, unsigned int batch_size) {
    m_iChannel = in_channel;
    m_iHeight = in_height;
    m_iWidth = in_width;

    m_iNumberOfClasses = output_count;
    m_uiNetworkBatchSize = batch_size;

    char cache_path[256];
    std::string basepath = weight_path.substr(0, weight_path.find_last_of('.'));
    sprintf(cache_path, "%s.%d.%d.%d.%u.engine", basepath.c_str(), m_iHeight, m_iWidth, m_iNumberOfClasses, m_uiNetworkBatchSize);
    
    printf(LOG_TRT "attempting to open cache file %s\n", cache_path);
    std::ifstream cache_file(cache_path, std::ios::binary);

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (!cache_file) {
        std::cout << "[" << PROJECT << "] Cache file is not found. Creating engine ..." << std::endl;
        std::map<std::string, nvinfer1::Weights> weightMap;
        if (loadWeights(weight_path, weightMap) != true){
            errorOccurred = true;
            return;
        }

        nvinfer1::IHostMemory *modelStream{nullptr};
        // Create builder
        nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);

        // Create model to populate the network, then set the outputs and create an engine
        nvinfer1::ICudaEngine *engine = CreateEngine(weightMap, m_uiNetworkBatchSize, builder,
                                                        nvinfer1::DataType::kFLOAT);
        if(engine == nullptr){
            std::cerr << "[" << PROJECT << "] Engine creation failed." << std::endl;
            errorOccurred = true;
            return;
        }
        else{
            std::cout << "[" << PROJECT << "] Engine creation is successful." << std::endl;
        }

        // Serialize the engine
        modelStream = engine->serialize();
        if(modelStream == nullptr){
            std::cerr << "[" << PROJECT << "] Engine serialization failed. Modelstream returned nullptr." << std::endl;
            errorOccurred = true;
            return;
        }
        // Close everything down
        engine->destroy();
        builder->destroy();

        std::ofstream p(cache_path);
        if (!p) {
            std::cerr << "[" << PROJECT << "] Could not open plan output file" << std::endl;
            errorOccurred = true;
            return;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        cache_file = std::ifstream(cache_path, std::ios::binary);
    }

    if (cache_file.good()) {
        cache_file.seekg(0, cache_file.end);
        size = cache_file.tellg();
        cache_file.seekg(0, cache_file.beg);
        trtModelStream = new char[size];
        cache_file.read(trtModelStream, size);
        cache_file.close();
    }
    else {
        std::cerr << "[" << PROJECT << "] Failed to open cache file." << std::endl;
        errorOccurred = true;
        return;
    }

    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if(m_runtime == nullptr){
        std::cerr << "[" << PROJECT << "] CreateInferRuntime failed." << std::endl;
        errorOccurred = true;
        return;
    }
    m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    if(m_engine == nullptr){
        std::cerr << "[" << PROJECT << "] DeserializeCudaEngine failed." << std::endl;
        errorOccurred = true;
        return;
    }
    m_context = m_engine->createExecutionContext();
    if(m_context == nullptr){
        std::cerr << "[" << PROJECT << "] CreateExecutionContext failed." << std::endl;
        errorOccurred = true;
        return;
    }
}

Resnet34::~Resnet34() {
    if (m_context != nullptr) { m_context->destroy(); }
    if (m_engine != nullptr) { m_engine->destroy(); }
    if (m_runtime != nullptr) { m_runtime->destroy(); }
    if (m_pNetInputGPU != nullptr) {
        cudaFree(m_pNetInputGPU);
        m_pNetInputGPU = nullptr;
    }
    if (m_pNetOutputGPU != nullptr) {
        cudaFree(m_pNetOutputGPU);
        m_pNetOutputGPU = nullptr;
    }
    if (m_pNetOutputCPU != nullptr) {
        cudaFreeHost(m_pNetOutputCPU);
        m_pNetOutputCPU = nullptr;
    }
    if (m_stream != NULL) { cudaStreamDestroy(m_stream); }
}

nvinfer1::ICudaEngine* Resnet34::CreateEngine(std::map<std::string, nvinfer1::Weights> weightMap, unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::DataType dt){
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    nvinfer1::ITensor* input = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims{3, {m_iChannel, m_iHeight, m_iWidth}});
    assert(input);

    nvinfer1::ITensor* base = resnet34(network, weightMap, input, m_iNumberOfClasses);

    base->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*base);
    
    // Build engine
    auto builder_config = builder->createBuilderConfig();
    builder_config->setMaxWorkspaceSize(1 << 20);
    builder->setMaxBatchSize(maxBatchSize);

    if (builder->platformHasFastFp16())
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *builder_config);

    network->destroy();

    // Release host memory
    for (auto& mem : weightMap){
        free((void*) (mem.second.values));
    }

    return engine;
}

bool Resnet34::AllocateMemory() {
    auto inputDims = GetTensorDims(INPUT_BLOB_NAME);
    auto outputDims = GetTensorDims(OUTPUT_BLOB_NAME);
    size_t inputSize = 1;
    size_t outSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) inputSize *= inputDims.d[i];
    for (int i = 0; i < outputDims.nbDims; i++) outSize *= outputDims.d[i];

    m_networkInputSize = m_uiNetworkBatchSize * inputSize * sizeof(float);
    m_networkOutputSize = m_uiNetworkBatchSize * outSize * sizeof(float);

    if ((m_networkInputSize == 0) || (m_networkOutputSize == 0)) {
        std::cerr << "[" << PROJECT << "] Invalid network size is encountered." << std::endl;
        return false;
    }

    CHECK(cudaMalloc((void**)&m_pNetInputGPU, m_networkInputSize));
    CHECK(cudaMalloc((void**)&m_pNetOutputGPU, m_networkOutputSize));
    CHECK(cudaMallocHost((void**)&m_pNetOutputCPU, m_networkOutputSize));

    return true;
}

const nvinfer1::Dims Resnet34::GetTensorDims(const char *name) {
    int bindingIndex = m_engine->getBindingIndex(name);
    if (bindingIndex != -1){
        return m_engine->getBindingDimensions(bindingIndex);
    }
    return nvinfer1::Dims{3, {0, 0, 0}};
}

bool Resnet34::Init() {
    if (errorOccurred) {
        return false;
    }
    if(AllocateMemory() == false){
        std::cerr << "[" << PROJECT << "] AllocateMemory failed." << std::endl;
        errorOccurred = true;
        return false;            
    }
    if (cudaStreamCreate(&m_stream) != cudaSuccess) {
        std::cerr << "[" << PROJECT << "] Stream creation failed." << std::endl;
        errorOccurred = true;
        return false;
    }
    m_bInitialized = true;
    return true;
}

bool Resnet34::DoInference(float* input) {
    if (!m_bInitialized || errorOccurred) {
        std::cerr << "[" << PROJECT << "] Error occurred before inference operation." << std::endl;
        return false;
    }
    if(m_engine->getNbBindings() != 2){
        std::cerr << "[" << PROJECT << "] Number of bindings for the engine is invalid" << std::endl;
        return false;
    }
    void *buffers[2];
    buffers[0] = m_pNetInputGPU;
    buffers[1] = m_pNetOutputGPU;
    CHECK(cudaMemcpyAsync(m_pNetInputGPU, input, m_networkInputSize, cudaMemcpyHostToDevice, m_stream));
    m_context->enqueue(m_uiNetworkBatchSize, buffers, m_stream, nullptr);
    CHECK(cudaMemcpyAsync(m_pNetOutputCPU, m_pNetOutputGPU, m_networkOutputSize, cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);
    return true;
}

