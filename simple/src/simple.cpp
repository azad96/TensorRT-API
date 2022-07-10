#include "simple.h"


SIMPLE::SIMPLE(std::vector<const char*> inputNames, std::vector<const char*> outputNames, std::string projectName, 
        std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU):
        TensorRT(inputNames, outputNames, projectName, inputDims, batchSize, outputToCPU)
{
}


bool SIMPLE::SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, unsigned int maxBatchSize, nvinfer1::DataType dt){
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    int channel = mInputDims[0][0];
    int height = mInputDims[0][1];
    int width = mInputDims[0][2];
    nvinfer1::ITensor* input = network->addInput(mInputNames[0], dt, nvinfer1::Dims{3, {channel, height, width}});
    assert(input);

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*input, 10, nvinfer1::DimsHW{3, 3}, weightMap["conv.weight"], weightMap["conv.bias"]);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});

    conv1->getOutput(0)->setName(mOutputNames[0]);
    network->markOutput(*conv1->getOutput(0));
    
    // Build engine
    auto builderConfig = builder->createBuilderConfig();
    builderConfig->setMaxWorkspaceSize(1 << 30);
    builder->setMaxBatchSize(maxBatchSize);

    if (builder->platformHasFastFp16())
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);

    // nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *builderConfig);

    // nvinfer1::IHostMemory* modelStream{nullptr};
    // modelStream = engine->serialize();

    nvinfer1::IHostMemory* modelStream = builder->buildSerializedNetwork(*network, *builderConfig);

    assert(modelStream != nullptr);
    std::ofstream p(cachePath);
    if (!p) {
        std::cerr << "[" << mProjectName << "] Could not open plan output file." << std::endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    // Close everything down
    // engine->destroy();
    network->destroy();
    builder->destroy();

    // Release host memory
    for (auto& mem : weightMap){
        free((void*) (mem.second.values));
    }

    return true;
}