#ifndef FEATURENET_H
#define FEATURENET_H

#include "tensorrt_base.h"

// stuff we know about the network
static const std::vector<const char*> INPUT_NAMES {"data"};
static const std::vector<const char*> OUTPUT_NAMES {"stage1", "stage2", "stage3"};

class FeatureNet: public TensorRT {
public:
    explicit FeatureNet(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
                std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU);
                
    void SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::IBuilder* &builder, 
                        nvinfer1::IBuilderConfig* &config, nvinfer1::IHostMemory* &modelStream, 
                        unsigned int maxBatchSize, nvinfer1::DataType dt);
};

#endif // FEATURENET_H

