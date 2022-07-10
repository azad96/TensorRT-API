//
// Created by azad on 20.04.2021.
//

#ifndef HTC_MASK_HEAD_H
#define HTC_MASK_HEAD_H

#include "NvInfer.h"
#include <map>
#include <vector>

class MaskHead {
public:
    explicit MaskHead(nvinfer1::INetworkDefinition* &network, std::map<std::string, nvinfer1::Weights>& weightMap);
    nvinfer1::ITensor* Run(nvinfer1::ITensor* mask_feats, nvinfer1::ITensor* &last_feat, int head_id, int class_number);
    // nvinfer1::ITensor* GetSegMasks(nvinfer1::ITensor* &mask_pred, nvinfer1::ITensor* &det_bboxes, std::vector<int> ori_shape);

private:
    // nvinfer1::ITensor* DoPasteMask(nvinfer1::ITensor* &masks, nvinfer1::ITensor* &bboxes, int img_h, int img_w);
    nvinfer1::INetworkDefinition *mpNetwork;
    std::map<std::string, nvinfer1::Weights> mWeightMap;
    int mConvLayerCount = 4;
    float mScaleFactor[4] = {1.0, 1.0, 1.0, 1.0};
    bool mRescale = true;
    

};

#endif //HTC_MASK_HEAD_H
