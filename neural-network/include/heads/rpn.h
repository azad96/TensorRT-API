//
// Created by botan on 14.12.2020.
//

#ifndef HTC_RPN_H
#define HTC_RPN_H

#include "NvInfer.h"
#include "yaml-cpp/yaml.h"


class Rpn {
public:
    explicit Rpn(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, int number_of_anchors);
    std::vector<nvinfer1::ITensor*> Run(std::vector<nvinfer1::ITensor *> &inputs,
                                        std::vector<nvinfer1::ITensor *> &multi_level_anchors);


private:
    std::vector<nvinfer1::ITensor*> SingleRun(nvinfer1::ITensor* input);

    nvinfer1::INetworkDefinition *mpNetwork;
    std::map<std::string, nvinfer1::Weights> mWeightMap;
    int mNmsPre = 1000;
    int mNumberOfAnchors;
    int mExplicitBatchOffset = 0;
    // std::vector<int> mScales = {1, 2, 4, 8, 16};
    // std::vector<int> mStrides = {4, 8, 16, 32, 64};
    // std::vector<float> mRatios = {0.33, 0.5, 1.0, 2.0, 3.0};

};

#endif //HTC_RPN_H
