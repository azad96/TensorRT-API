#ifndef SIMPLE_H
#define SIMPLE_H

#include "tensorrt_base.h"

// stuff we know about the network
static const std::vector<const char*> INPUT_NAMES {"data"};
static const std::vector<const char*> OUTPUT_NAMES {"prob"};
static const char* PROJECT_NAME = "SIMPLE";

class SIMPLE: public TensorRT {
public:
    explicit SIMPLE(std::vector<const char*> inputNames, std::vector<const char*> outputNames, std::string projectName, 
                std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU);
    bool SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, unsigned int maxBatchSize, nvinfer1::DataType dt);
};

#endif // SIMPLE_H

