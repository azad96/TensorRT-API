#ifndef HRNET_BACKBONE_H
#define HRNET_BACKBONE_H

#include "NvInfer.h"
#include "yaml-cpp/yaml.h"

    
class HrnetBackbone {
public:
    explicit HrnetBackbone(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, YAML::Node &config);
    
    std::vector<nvinfer1::ITensor*> Run(nvinfer1::ITensor* input);

private:
    nvinfer1::ITensor* AddBottleneck(nvinfer1::ITensor &input, std::string lname,  int planes , bool hasDownsample);

    nvinfer1::ITensor* MakeLayer(nvinfer1::ITensor &input, std::string lname, std::string blockType, 
                                int inplanes, int planes , int blocks, int stride=1);

    std::vector<nvinfer1::ITensor*> MakeTransitionLayer(std::vector<nvinfer1::ITensor *> &inputs, std::string lname, 
                                                        std::vector<int> numChannelsPreLayer, std::vector<int> numChannelsCurLayer);

    nvinfer1::ITensor* AddBasicBlock(nvinfer1::ITensor* input, std::string lname, nvinfer1::DimsHW stride, int planes, bool hasDownsample=false);

    std::vector<std::vector<std::vector<std::map<std::string, std::string>>>> MakeFuseLayers(std::string lname, int numBranches, std::vector<int> numInChannels, int fuseLayerCount);

    nvinfer1::ITensor* AddConfigLayer(nvinfer1::ITensor* input,  std::vector<std::map<std::string, std::string>> seqLayerInfo);

    std::vector<nvinfer1::ITensor*> AddHighResolutionModule(std::vector<nvinfer1::ITensor *> inputs, std::string lname, int numBranches, std::string blockType, 
                                                            std::vector<int> numBlocks, std::vector<int> numInChannels, std::vector<int> numChannels, 
                                                            std::string fuseMethod, bool multiScaleOutput = true, int stride = 1);

    std::vector<nvinfer1::ITensor*> MakeStage(std::vector<nvinfer1::ITensor*> inputs, std::string lname, std::vector<int> numInChannels, 
                                            YAML::Node &stageConfig, bool multiScaleOutput = true);

    nvinfer1::INetworkDefinition *mpNetwork;
    std::map<std::string, nvinfer1::Weights> mWeightMap;
    YAML::Node mConfig;
    int mBottleneckExpansion=4;
    int mBasicExpansion=1;
};

#endif 
