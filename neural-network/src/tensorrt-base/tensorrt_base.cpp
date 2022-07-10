#include "tensorrt_base.h"


TensorRT::TensorRT(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
                    std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU) {
    mInputNames = inputNames;
    mOutputNames = outputNames;
    mInputDims = inputDims;
    mNetworkBatchSize = batchSize;
    mOutputToCPU = outputToCPU;

    for (int i = 0; i < mInputNames.size(); i++)
        mInputPointersGPU.push_back(nullptr);

    for (int i = 0; i < mOutputNames.size(); i++){
        mOutputPointersGPU.push_back(nullptr);
        mOutputPointersCPU.push_back(nullptr);
    }
}


TensorRT::~TensorRT() {
    cudaStreamDestroy(mStream);

    for (int i = 0; i < mInputPointersGPU.size(); i++){
        if (mInputPointersGPU[i] != nullptr) 
            cudaFree(mInputPointersGPU[i]);
    }

    for (int i = 0; i < mOutputPointersGPU.size(); i++){
        if (mOutputPointersGPU[i] != nullptr) 
            cudaFree(mOutputPointersGPU[i]);
    }

    for (int i = 0; i < mOutputPointersCPU.size(); i++){
        if (mOutputPointersCPU[i] != nullptr) 
            delete [] mOutputPointersCPU[i];
            // cudaFree(mOutputPointersCPU[i]);
    }

    mInputPointersGPU.clear();
    mOutputPointersGPU.clear();
    mOutputPointersCPU.clear();

    if (mpContext != nullptr) mpContext->destroy();
    if (mpEngine != nullptr) mpEngine->destroy();
    if (mpRuntime != nullptr) mpRuntime->destroy();
}


const nvinfer1::Dims TensorRT::GetTensorDims(const char *name) {
    int bindingIndex = mpEngine->getBindingIndex(name);
    assert(bindingIndex != -1);
    return mpEngine->getBindingDimensions(bindingIndex);
}


const nvinfer1::DataType TensorRT::GetTensorDataType(const char *name) {
    int bindingIndex = mpEngine->getBindingIndex(name);
    assert(bindingIndex != -1);
    return mpEngine->getBindingDataType(bindingIndex);
}


const size_t TensorRT::GetTensorDataTypeSize(const char *name) {
    auto outputDataType = GetTensorDataType(name);

    size_t data_type_size = 1; 

    if (outputDataType == nvinfer1::DataType::kFLOAT){
        data_type_size = sizeof(float);
    }
    else if (outputDataType == nvinfer1::DataType::kHALF){
        std::cerr << "DataType::kHALF is not implemented for AllocateMemory. DataType::kFLOAT will be used instead." << std::endl;
        data_type_size = sizeof(float);
    }
    else if (outputDataType == nvinfer1::DataType::kINT8){
        data_type_size = sizeof(int8_t);
    }
    else if (outputDataType == nvinfer1::DataType::kINT32){
        data_type_size = sizeof(int32_t);
    }
    else if (outputDataType == nvinfer1::DataType::kBOOL){
        data_type_size = sizeof(bool);
    }
    else{
        throw std::invalid_argument( "Invalid DataType is encountered" );
    }

    return data_type_size;
}


void TensorRT::DeserializeEngine(std::ifstream& cacheFile){
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (cacheFile.good()) {
        cacheFile.seekg(0, cacheFile.end);
        size = cacheFile.tellg();
        cacheFile.seekg(0, cacheFile.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        cacheFile.read(trtModelStream, size);
        cacheFile.close();
    }
    else throw std::runtime_error("Failed to open cache file.");
    

    mpRuntime = nvinfer1::createInferRuntime(gLogger);
    assert(mpRuntime != nullptr);

    mpEngine = mpRuntime->deserializeCudaEngine(trtModelStream, size);
    assert(mpEngine != nullptr);

    mpContext = mpEngine->createExecutionContext();
    assert(mpContext != nullptr);
}


void TensorRT::PrepareInference(std::string &weightPath){
    char cachePath[128];
    std::string basepath = weightPath.substr(0, weightPath.find_last_of('.'));
    sprintf(cachePath, "%s.engine", basepath.c_str());
    
    std::ifstream cacheFile(cachePath, std::ios::binary);

    if(!cacheFile.good()){
        std::cout << "Cache file is not found. Creating engine..." << std::endl;

        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        nvinfer1::IHostMemory* modelStream = nullptr;

        std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(weightPath);
        SerializeEngine(cachePath, weightMap, builder, config, modelStream, mNetworkBatchSize, nvinfer1::DataType::kFLOAT);
        cacheFile = std::ifstream(cachePath, std::ios::binary);
        std::cout << "Engine serialization is successful." << std::endl;

        builder->destroy();
        modelStream->destroy();
    }
    else
        std::cout << "Cache file is being used..." << std::endl;

    DeserializeEngine(cacheFile);
}


void TensorRT::Init() {
    for (int i = 0; i < mInputNames.size(); i++){
        auto inputDims = GetTensorDims(mInputNames[i]);
        size_t inputSize = 1 * mNetworkBatchSize * GetTensorDataTypeSize(mInputNames[i]);

        for (int j = 0; j < inputDims.nbDims; j++) 
            inputSize *= inputDims.d[j];

        assert(inputSize != 0);
        mNetworkInputSizes.push_back(inputSize);
        CHECK(cudaMalloc((void**)&mInputPointersGPU[i], inputSize));
    }

    for (int i = 0; i < mOutputNames.size(); i++){
        auto outputDims = GetTensorDims(mOutputNames[i]);
        size_t outputSize = 1 * mNetworkBatchSize * GetTensorDataTypeSize(mOutputNames[i]);

        for (int j = 0; j < outputDims.nbDims; j++) 
            outputSize *= outputDims.d[j];

        assert(outputSize != 0);
        mNetworkOutputSizes.push_back(outputSize);
        CHECK(cudaMalloc((void**)&mOutputPointersGPU[i], outputSize));
        if (mOutputToCPU)
            mOutputPointersCPU[i] = new float[outputSize];
            // CHECK(cudaMallocHost((void**)&mOutputPointersCPU[i], outputSize));
    }
    CHECK(cudaStreamCreate(&mStream));
    // if (cudaStreamCreate(&mStream) != cudaSuccess)
    //     throw std::runtime_error("cudaStreamCreate failed.");
}


void TensorRT::DoInference(int batchSize, float* inputs[]) {
    int inputBindings = mInputPointersGPU.size();
    int outputBindings = mOutputPointersGPU.size();
    
    assert(mpEngine->getNbBindings() == inputBindings + outputBindings);

    void *buffers[inputBindings + outputBindings];
    for (int i = 0; i < inputBindings;  i++) buffers[i] = mInputPointersGPU[i];
    for (int i = 0; i < outputBindings; i++) buffers[inputBindings + i] = mOutputPointersGPU[i];
    
    for (int i = 0; i < inputBindings; i++)
        CHECK(cudaMemcpyAsync(mInputPointersGPU[i], inputs[i], mNetworkInputSizes[i], cudaMemcpyHostToDevice, mStream));

    mpContext->enqueue(batchSize, buffers, mStream, nullptr);

    if (mOutputToCPU){
        for (int i = 0; i < outputBindings; i++)
            CHECK(cudaMemcpyAsync(mOutputPointersCPU[i], mOutputPointersGPU[i], mNetworkOutputSizes[i], cudaMemcpyDeviceToHost, mStream));
    }

    cudaStreamSynchronize(mStream);
}
