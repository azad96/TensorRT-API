#include "featurenet.h"


nvinfer1::ITensor* conv_batch_relu(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor* input, \
                                    int outChannel, int kernel, int stride, int padding, std::string lname) 
{
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto conv1 = network->addConvolutionNd(*input, outChannel, nvinfer1::DimsHW{kernel, kernel}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv1->setPaddingNd(nvinfer1::DimsHW{padding, padding});

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);

    auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);
    
    return relu1->getOutput(0);
}


FeatureNet::FeatureNet(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
        std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU):
        TensorRT(inputNames, outputNames, inputDims, batchSize, outputToCPU)
{
}


void FeatureNet::SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::IBuilder* &builder, 
                                nvinfer1::IBuilderConfig* &config, nvinfer1::IHostMemory* &modelStream, unsigned int maxBatchSize, nvinfer1::DataType dt)
{
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    int view = mInputDims[0][0];
    int channel = mInputDims[0][1];
    int height = mInputDims[0][2];
    int width = mInputDims[0][3];

    nvinfer1::ITensor* input = network->addInput(mInputNames[0], dt, nvinfer1::Dims{4, {view, channel, height, width}});
    assert(input);

    // conv0
    auto conv0_block1 = conv_batch_relu(network, weightMap, input, 8, 3, 1, 1, "feature_net.conv0.0");
    auto conv_stage3 = conv_batch_relu(network, weightMap, conv0_block1, 8, 3, 1, 1, "feature_net.conv0.1");
    // conv0

    // conv1
    auto conv1_block1 = conv_batch_relu(network, weightMap, conv_stage3, 16, 5, 2, 2, "feature_net.conv1.0");
    auto conv1_block2 = conv_batch_relu(network, weightMap, conv1_block1, 16, 3, 1, 1, "feature_net.conv1.1");
    auto conv_stage2 = conv_batch_relu(network, weightMap, conv1_block2, 16, 3, 1, 1, "feature_net.conv1.2");
    // conv1

    // conv2
    auto conv2_block1 = conv_batch_relu(network, weightMap, conv_stage2, 32, 5, 2, 2, "feature_net.conv2.0");
    auto conv2_block2 = conv_batch_relu(network, weightMap, conv2_block1, 32, 3, 1, 1, "feature_net.conv2.1");
    auto conv_stage1 = conv_batch_relu(network, weightMap, conv2_block2, 32, 3, 1, 1, "feature_net.conv2.2");
    // conv2

    // stage1
    auto res_stage1 = network->addConvolutionNd(*conv_stage1, 32, nvinfer1::DimsHW{1, 1}, weightMap["feature_net.out.stage1.weight"], emptywts);
    assert(res_stage1);
    res_stage1->setStrideNd(nvinfer1::DimsHW{1, 1});

    res_stage1->getOutput(0)->setName(mOutputNames[0]);
    network->markOutput(*res_stage1->getOutput(0));
    // stage1

    // stage2
    auto interpolate_conv_stage1 = network->addResize(*conv_stage1);
    assert(interpolate_conv_stage1);

    float scales[4] = {1, 1, 2, 2};
    interpolate_conv_stage1->setScales(scales, 4);
    interpolate_conv_stage1->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    interpolate_conv_stage1->setAlignCorners(false);

    auto skip_stage2 = network->addConvolutionNd(*conv_stage2, 32, nvinfer1::DimsHW{1, 1}, weightMap["feature_net.skip.stage2.weight"], weightMap["feature_net.skip.stage2.bias"]);
    assert(skip_stage2);
    skip_stage2->setStrideNd(nvinfer1::DimsHW{1, 1});

    auto inter_stage2 = network->addElementWise(*interpolate_conv_stage1->getOutput(0), *skip_stage2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    auto res_stage2 = network->addConvolutionNd(*inter_stage2->getOutput(0), 16, nvinfer1::DimsHW{3, 3}, weightMap["feature_net.out.stage2.weight"], emptywts);
    assert(res_stage2);
    res_stage2->setStrideNd(nvinfer1::DimsHW{1, 1});
    res_stage2->setPaddingNd(nvinfer1::DimsHW{1, 1});

    res_stage2->getOutput(0)->setName(mOutputNames[1]);
    network->markOutput(*res_stage2->getOutput(0));
    // stage2

    // stage3
    auto interpolate_inter_stage2 = network->addResize(*inter_stage2->getOutput(0));
    assert(interpolate_inter_stage2);

    interpolate_inter_stage2->setScales(scales, 4);
    interpolate_inter_stage2->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    interpolate_inter_stage2->setAlignCorners(false);

    auto skip_stage3 = network->addConvolutionNd(*conv_stage3, 32, nvinfer1::DimsHW{1, 1}, weightMap["feature_net.skip.stage3.weight"], weightMap["feature_net.skip.stage3.bias"]);
    assert(skip_stage3);
    skip_stage3->setStrideNd(nvinfer1::DimsHW{1, 1});

    auto inter_stage3 = network->addElementWise(*interpolate_inter_stage2->getOutput(0), *skip_stage3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    auto res_stage3 = network->addConvolutionNd(*inter_stage3->getOutput(0), 8, nvinfer1::DimsHW{3, 3}, weightMap["feature_net.out.stage3.weight"], emptywts);
    assert(res_stage3);
    res_stage3->setStrideNd(nvinfer1::DimsHW{1, 1});
    res_stage3->setPaddingNd(nvinfer1::DimsHW{1, 1});

    res_stage3->getOutput(0)->setName(mOutputNames[2]);
    network->markOutput(*res_stage3->getOutput(0));
    // stage3

    // Build engine
    config->setMaxWorkspaceSize(1 << 30);
    builder->setMaxBatchSize(maxBatchSize);

    if (builder->platformHasFastFp16()){
        std::cout << "FP16 is supported" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    modelStream = builder->buildSerializedNetwork(*network, *config);
    assert(modelStream != nullptr);
    
    // nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // // Serialize the engine
    // modelStream = engine->serialize();
    // assert(modelStream != nullptr);

    std::ofstream plan_file(cachePath, std::ios::binary);
    if (!plan_file) 
        throw std::runtime_error("Could not open plan output file.");
    plan_file.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    network->destroy();

    // Release host memory
    for (auto& mem : weightMap){
        free((void*) (mem.second.values));
    }
}