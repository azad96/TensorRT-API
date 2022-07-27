#include "cost_reg_net.h"


nvinfer1::ITensor* conv3d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor* input, \
                                    int outChannel, int kernel, int stride, int padding, std::string lname) 
{
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto conv1 = network->addConvolutionNd(*input, outChannel, nvinfer1::Dims{3, {kernel, kernel, kernel}}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::Dims{3, {stride, stride, stride}});
    conv1->setPaddingNd(nvinfer1::Dims{3, {padding, padding, padding}});

    auto bn1 = addBatchNormNd(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);
    assert(bn1);

    auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);
    
    return relu1->getOutput(0);
}


nvinfer1::ITensor* deconv3d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor* input, \
                                    int outChannel, int kernel, int stride, int padding, int out_padding, std::string lname) 
{
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto deconv1 = network->addDeconvolutionNd(*input, outChannel, nvinfer1::Dims{3, {kernel, kernel, kernel}}, weightMap[lname + ".conv.weight"], emptywts);
    assert(deconv1);
    deconv1->setStrideNd(nvinfer1::Dims{3, {stride, stride, stride}});
    deconv1->setPaddingNd(nvinfer1::Dims{3, {padding, padding, padding}});
    
    auto dims = deconv1->getOutput(0)->getDimensions().d;

    const float fill_value[1] = {0};
    auto fill_value_layer = network->addConstant(nvinfer1::Dims{0, {0}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, fill_value, 1});

    auto slice1 = network->addSlice(*deconv1->getOutput(0), 
                                    nvinfer1::Dims{4, {0, 0, 0, 0}},
                                    nvinfer1::Dims{4, {dims[0], dims[1]+1, dims[2]+1, dims[3]+1}},
                                    nvinfer1::Dims{4, {1, 1, 1, 1}});
    assert(slice1);
    
    // slice1->setMode(nvinfer1::SliceMode::kWRAP);
    slice1->setMode(nvinfer1::SliceMode::kFILL);
    slice1->setInput(4, *fill_value_layer->getOutput(0));

    // auto pad1 = network->addPaddingNd(*deconv1->getOutput(0), nvinfer1::Dims2{0,0}, nvinfer1::Dims2{1,1});
    // assert(pad1);

    // int dim1 = pad1->getOutput(0)->getDimensions().d[0];
    // int dim3 = pad1->getOutput(0)->getDimensions().d[2];
    // int dim4 = pad1->getOutput(0)->getDimensions().d[3];

    // int zero_arr_size = dim1 * dim3 * dim4;
    // float *zero_arr = new float[zero_arr_size];
    // std::fill_n(zero_arr, zero_arr_size, 0.0f);
    // weightMap["new_zero_arr"] = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, zero_arr, zero_arr_size};
    // auto const1 = network->addConstant(nvinfer1::Dims{4, {dim1, 1, dim3, dim4}}, weightMap["new_zero_arr"]);
    
    // nvinfer1::ITensor* arrays_to_concat[] = {pad1->getOutput(0), const1->getOutput(0)};
    // auto concat1 = network->addConcatenation(arrays_to_concat, 2);
    // assert(concat1);
    // concat1->setAxis(1);

    auto bn1 = addBatchNormNd(network, weightMap, *slice1->getOutput(0), lname + ".bn", 1e-5);
    assert(bn1);

    auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);
    
    return relu1->getOutput(0);
}


nvinfer1::ITensor* costRegBlock(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap,
                                 nvinfer1::ITensor* input, int costVolumeBaseChannel, std::string lname) 
{
    auto conv0 = conv3d(network, weightMap, input, costVolumeBaseChannel, 3, 1, 1, lname + ".conv0");
    auto conv1 = conv3d(network, weightMap, conv0, costVolumeBaseChannel*2, 3, 2, 1, lname + ".conv1");
    auto conv2 = conv3d(network, weightMap, conv1, costVolumeBaseChannel*2, 3, 1, 1, lname + ".conv2");
    auto conv3 = conv3d(network, weightMap, conv2, costVolumeBaseChannel*4, 3, 2, 1, lname + ".conv3");
    auto conv4 = conv3d(network, weightMap, conv3, costVolumeBaseChannel*4, 3, 1, 1, lname + ".conv4");
    auto conv5 = conv3d(network, weightMap, conv4, costVolumeBaseChannel*8, 3, 2, 1, lname + ".conv5");
    auto conv6 = conv3d(network, weightMap, conv5, costVolumeBaseChannel*8, 3, 1, 1, lname + ".conv6");
    auto conv7 = deconv3d(network, weightMap, conv6, costVolumeBaseChannel*4, 3, 2, 1, 1, lname + ".conv7");
    auto ew1 = network->addElementWise(*conv4, *conv7, nvinfer1::ElementWiseOperation::kSUM);
    auto conv9 = deconv3d(network, weightMap, ew1->getOutput(0), costVolumeBaseChannel*2, 3, 2, 1, 1, lname + ".conv9");
    auto ew2 = network->addElementWise(*conv2, *conv9, nvinfer1::ElementWiseOperation::kSUM);
    auto conv11 = deconv3d(network, weightMap, ew2->getOutput(0), costVolumeBaseChannel, 3, 2, 1, 1, lname + ".conv11");
    auto ew3 = network->addElementWise(*conv0, *conv11, nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto prob = network->addConvolutionNd(*ew3->getOutput(0), 1, nvinfer1::Dims{3, {3, 3, 3}}, weightMap[lname + ".prob.weight"], emptywts);
    assert(prob);
    prob->setStrideNd(nvinfer1::Dims{3, {1, 1, 1}});
    prob->setPaddingNd(nvinfer1::Dims{3, {1, 1, 1}});

    return prob->getOutput(0);
}


CostRegNet::CostRegNet(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
        std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU):
        TensorRT(inputNames, outputNames, inputDims, batchSize, outputToCPU)
{
}


void CostRegNet::SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::IBuilder* &builder, 
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

    auto stage1 = costRegBlock(network, weightMap, input, 8, "cost_regularization_net.stage1");


    stage1->setName(mOutputNames[0]);
    network->markOutput(*stage1);

    // tmp->getOutput(0)->setName(mOutputNames[0]);
    // network->markOutput(*tmp->getOutput(0));

    // Build engine
    config->setMaxWorkspaceSize(1 << 30);
    builder->setMaxBatchSize(maxBatchSize);

    if (builder->platformHasFastFp16()){
        std::cout << "FP16 is supported" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    modelStream = builder->buildSerializedNetwork(*network, *config);
    assert(modelStream != nullptr);
    
    std::ofstream plan_file(cachePath, std::ios::binary);
    if (!plan_file) 
        throw std::runtime_error("Could not open plan output file.");
    plan_file.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    network->destroy();

    // Release host memory
    for (auto &mem : weightMap){   
        if (mem.first.rfind("new_", 0) == 0)
        //     delete[] (float*) mem.second.values;
        // else
            free((void *)(mem.second.values));
    }
}