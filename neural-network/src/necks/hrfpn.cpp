//
// Created by root on 11.12.2020.
//

#include "hrfpn.h"
#include "common.h"


Hrfpn::Hrfpn(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap)
{
    mpNetwork = network;
    mWeightMap = weightMap;
}

std::vector<nvinfer1::ITensor*> Hrfpn::InterpolateInput(std::vector<nvinfer1::ITensor *> &inputs)
{
    int num_inputs = inputs.size();

    std::vector<nvinfer1::ITensor*> outs;
    outs.push_back(inputs[0]);

    for(int i=1;i<num_inputs; i++)
    {
        nvinfer1::IResizeLayer *interpolate = mpNetwork->addResize(*inputs[i]);
        assert(interpolate);

        auto dims = inputs[i]->getDimensions();
        if (mpNetwork->hasImplicitBatchDimension()){
            int channel_num = dims.d[0];
            int height_in = dims.d[1];
            int width_in = dims.d[2];

            int height_output = int(height_in * pow(2,i));
            int width_output = int(width_in * pow(2,i));

            interpolate->setOutputDimensions(nvinfer1::Dims{3, {channel_num, height_output, width_output}});
        }
        else{
            int N = dims.d[0];
            int channel_num = dims.d[1];
            int height_in = dims.d[2];
            int width_in = dims.d[3];

            int height_output = int(height_in * pow(2,i));
            int width_output = int(width_in * pow(2,i));

            interpolate->setOutputDimensions(nvinfer1::Dims{4, {N, channel_num, height_output, width_output}});
        }

        interpolate->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
        interpolate->setAlignCorners(true);
        outs.push_back(interpolate->getOutput(0));
    }
    return outs;
}

std::vector<nvinfer1::ITensor*> Hrfpn::PoolInput(nvinfer1::ITensor* &input, int num_outs)
{

    std::vector<nvinfer1::ITensor*> outs;
    outs.push_back(input);

    for(int i=1;i<num_outs; i++)
    {
        int kernel_size = int(pow(2,i));
        int stride = int(pow(2,i));

        nvinfer1::IPoolingLayer* pooling_layer = mpNetwork->addPooling(*input,nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{kernel_size, kernel_size});
        assert(pooling_layer);
        pooling_layer->setStride(nvinfer1::DimsHW{stride, stride});
        outs.push_back(pooling_layer->getOutput(0));

    }
    return outs;
}

std::vector<nvinfer1::ITensor*> Hrfpn::Run(std::vector<nvinfer1::ITensor *> &inputs)
{
    int num_outs = 5;
    int out_channels = 256;

    std::vector<nvinfer1::ITensor*> out = InterpolateInput(inputs);
    //return out;


    nvinfer1::ITensor* interpolated_inputs[] = {out[0], out[1], out[2], out[3]};

    nvinfer1::IConcatenationLayer *concat = mpNetwork->addConcatenation(interpolated_inputs, out.size());
    assert(concat);
    if (mpNetwork->hasImplicitBatchDimension()){
        concat->setAxis(0);
    }
    else{
        concat->setAxis(1);
    }

    //nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* reduction_conv = mpNetwork->addConvolution(*concat->getOutput(0),out_channels,nvinfer1::DimsHW{1, 1},
                                                                            mWeightMap["neck.reduction_conv.conv.weight"],
                                                                            mWeightMap["neck.reduction_conv.conv.bias"]);

    assert(reduction_conv);
    reduction_conv->setStride(nvinfer1::DimsHW{1, 1});

    nvinfer1::ITensor* reducted_feature = reduction_conv->getOutput(0);

    std::vector<nvinfer1::ITensor*> pooled_features = PoolInput(reducted_feature,num_outs);

    std::vector<nvinfer1::ITensor*> hrfpn_output;

    for(int i=0; i<num_outs; i++)
    {
        nvinfer1::IConvolutionLayer* conv = mpNetwork->addConvolution(*pooled_features[i],out_channels,nvinfer1::DimsHW{3, 3},
                                                                                mWeightMap["neck.fpn_convs." + std::to_string(i) + ".conv.weight"],
                                                                                mWeightMap["neck.fpn_convs." + std::to_string(i) + ".conv.bias"]);
        conv->setStride(nvinfer1::DimsHW(1,1));
        conv->setPadding(nvinfer1::DimsHW(1,1));
        hrfpn_output.push_back(conv->getOutput(0));

    }

    return hrfpn_output;
}