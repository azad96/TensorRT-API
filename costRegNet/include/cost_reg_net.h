#ifndef COST_REG_NET_H
#define COST_REG_NET_H

#include "tensorrt_base.h"


class CostRegNet: public TensorRT {
public:
    explicit CostRegNet(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
                std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU);
                
    void SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::IBuilder* &builder, 
                        nvinfer1::IBuilderConfig* &config, nvinfer1::IHostMemory* &modelStream, 
                        unsigned int maxBatchSize, nvinfer1::DataType dt);
private:
};

#endif // COST_REG_NET_H

