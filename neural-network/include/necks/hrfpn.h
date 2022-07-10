//
// Created by root on 11.12.2020.
//

#ifndef HTC_HRFPN_H
#define HTC_HRFPN_H

#endif //HTC_HRFPN_H

#include "NvInfer.h"
#include "yaml-cpp/yaml.h"


class Hrfpn {
public:
    explicit Hrfpn(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap);

    std::vector<nvinfer1::ITensor*> Run(std::vector<nvinfer1::ITensor *> &inputs);

private:
    std::vector<nvinfer1::ITensor*> InterpolateInput(std::vector<nvinfer1::ITensor *> &inputs);
    std::vector<nvinfer1::ITensor*> PoolInput(nvinfer1::ITensor* &input, int num_outs);

    nvinfer1::INetworkDefinition *mpNetwork;
    std::map<std::string, nvinfer1::Weights> mWeightMap;

};

