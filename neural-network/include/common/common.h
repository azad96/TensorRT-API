#ifndef TRT_COMMON_H
#define TRT_COMMON_H

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)


std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);

#endif // TRT_COMMON_H
