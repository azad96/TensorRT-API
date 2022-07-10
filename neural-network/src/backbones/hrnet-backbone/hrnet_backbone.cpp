
#include "common.h"
#include "hrnet_backbone.h"


HrnetBackbone::HrnetBackbone(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, YAML::Node &config):
    mpNetwork(network), mWeightMap(weightMap), mConfig(config)
{
}


nvinfer1::ITensor* HrnetBackbone::AddBottleneck(nvinfer1::ITensor &input, std::string lname, int planes, bool hasDownsample) 
{
    int stride=1;
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer* conv1 = mpNetwork->addConvolution(input, planes, nvinfer1::DimsHW{1, 1}, mWeightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(mpNetwork, mWeightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    nvinfer1::IActivationLayer* relu1 = mpNetwork->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IConvolutionLayer *conv2 = mpNetwork->addConvolution(*relu1->getOutput(0), planes, nvinfer1::DimsHW{3, 3}, mWeightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(nvinfer1::DimsHW{stride, stride});
    conv2->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn2 = addBatchNorm2d(mpNetwork, mWeightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    nvinfer1::IActivationLayer* relu2 = mpNetwork->addActivation(*bn2->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);

    nvinfer1::IConvolutionLayer *conv3 = mpNetwork->addConvolution(*relu2->getOutput(0), planes * mBottleneckExpansion, nvinfer1::DimsHW{1, 1}, mWeightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    nvinfer1::IScaleLayer* bn3 = addBatchNorm2d(mpNetwork, mWeightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    nvinfer1::IElementWiseLayer* ew1;
    if (hasDownsample)
    {
        nvinfer1::IConvolutionLayer *conv4 = mpNetwork->addConvolution(input, planes * mBottleneckExpansion, nvinfer1::DimsHW{1, 1}, mWeightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(nvinfer1::DimsHW{stride, stride});

        nvinfer1::IScaleLayer* bn4 = addBatchNorm2d(mpNetwork, mWeightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = mpNetwork->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }
    else
    {
        ew1 = mpNetwork->addElementWise(input, *bn3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }
    nvinfer1::IActivationLayer* relu3 = mpNetwork->addActivation(*ew1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu3);
    return relu3->getOutput(0);
}


nvinfer1::ITensor* HrnetBackbone::MakeLayer(nvinfer1::ITensor &input, std::string lname, std::string blockType, int inplanes, int planes , int blocks, int stride)
{
    if (blockType == "BOTTLENECK")
    {
        bool has_downsample = false;
        if(stride != 1 || inplanes != planes * mBottleneckExpansion)
        {
            has_downsample = true;
        }
        nvinfer1::ITensor*  block_pointer_array[blocks]; 

        for (int block_it=0; block_it < blocks; block_it++)
        {
            if(block_it==0)
            {
                block_pointer_array[block_it] = AddBottleneck(input, lname + "." + std::to_string(block_it) + ".", planes, has_downsample);
            }
            else
            {
                block_pointer_array[block_it] = AddBottleneck(*block_pointer_array[block_it - 1], lname + "." + std::to_string(block_it) + ".", planes, false);
            }
        }

        return block_pointer_array[blocks-1] ;
    }
    else
    {
        return nullptr;
    }
}


std::vector<nvinfer1::ITensor *> HrnetBackbone::MakeTransitionLayer(std::vector<nvinfer1::ITensor *> &inputs, std::string lname, std::vector<int> numChannelsPreLayer, std::vector<int> numChannelsCurLayer)
{
    int num_branches_cur = numChannelsCurLayer.size();
    int num_branches_pre = numChannelsPreLayer.size();
    std::vector<nvinfer1::ITensor*> x_list;

    for (int i = 0; i < num_branches_cur; i++)
    {
        if (i < num_branches_pre){
            if (numChannelsCurLayer[i] != numChannelsPreLayer[i]){
                nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

                nvinfer1::IConvolutionLayer *conv1 = mpNetwork->addConvolution(*inputs[inputs.size()-1], numChannelsCurLayer[i], nvinfer1::DimsHW{3, 3}, mWeightMap[lname + "." + std::to_string(i) + ".0.weight"], emptywts);
                assert(conv1);
                conv1->setStride(nvinfer1::DimsHW{1, 1});
                conv1->setPadding(nvinfer1::DimsHW{1, 1});
                
                nvinfer1::IScaleLayer *bn1 = addBatchNorm2d(mpNetwork, mWeightMap, *conv1->getOutput(0), lname + "." + std::to_string(i) + ".1", 1e-5);

                nvinfer1::IActivationLayer* relu1 = mpNetwork->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
                assert(relu1);

                x_list.push_back(relu1->getOutput(0));
            }
            else{
                x_list.push_back(inputs[i]);
            }
        }
        else
        {
            nvinfer1::ITensor *res = inputs[inputs.size()-1];
            for (int j = 0; j < i+1-num_branches_pre; j++){
                int inchannels = numChannelsPreLayer[num_branches_pre-1];
                int outchannels = j == i-num_branches_pre ? numChannelsCurLayer[i] : inchannels;

                nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
                nvinfer1::IConvolutionLayer *conv1 = mpNetwork->addConvolution(*res, outchannels, nvinfer1::DimsHW{3, 3}, mWeightMap[lname + "." + std::to_string(i) + "."+ std::to_string(j) +".0.weight"], emptywts);
                assert(conv1);
                conv1->setStride(nvinfer1::DimsHW{2, 2});
                conv1->setPadding(nvinfer1::DimsHW{1, 1});

                nvinfer1::IScaleLayer *bn1 = addBatchNorm2d(mpNetwork, mWeightMap, *conv1->getOutput(0), lname + "." + std::to_string(i) + "." + std::to_string(j) + ".1", 1e-5);

                nvinfer1::IActivationLayer* relu1 = mpNetwork->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
                assert(relu1);
                res = relu1->getOutput(0);
            }
                x_list.push_back(res);
        }

    }
    return x_list;
}


nvinfer1::ITensor* HrnetBackbone::AddBasicBlock(nvinfer1::ITensor* input, std::string lname, nvinfer1::DimsHW stride, int planes, bool hasDownsample) {
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer* conv1 = mpNetwork->addConvolution(*input, planes, nvinfer1::DimsHW{3, 3}, mWeightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(stride);
    conv1->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(mpNetwork, mWeightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    nvinfer1::IActivationLayer* relu1 = mpNetwork->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IConvolutionLayer* conv2 = mpNetwork->addConvolution(*relu1->getOutput(0), planes, nvinfer1::DimsHW{3, 3}, mWeightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn2 = addBatchNorm2d(mpNetwork, mWeightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    nvinfer1::IElementWiseLayer* ew1;
    if (hasDownsample) {
        nvinfer1::IConvolutionLayer* conv3 = mpNetwork->addConvolution(*input, planes, nvinfer1::DimsHW{1, 1}, mWeightMap[lname + ".downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStride(stride);
        nvinfer1::IScaleLayer* bn3 = addBatchNorm2d(mpNetwork, mWeightMap, *conv3->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = mpNetwork->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    } else {
        ew1 = mpNetwork->addElementWise(*input, *bn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }

    nvinfer1::IActivationLayer* relu2 = mpNetwork->addActivation(*ew1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);
    return relu2->getOutput(0);
}


std::vector<std::vector<std::vector<std::map<std::string, std::string>>>> HrnetBackbone::MakeFuseLayers(std::string lname, int numBranches, std::vector<int> numInChannels, int fuseLayerCount)
{
    std::vector<std::vector<std::vector<std::map<std::string, std::string>>>> fuse_layers_module_list;

    for (int i=0 ; i<fuseLayerCount ; i++)
    {
        std::vector<std::vector<std::map<std::string, std::string>>> fuse_layer_module_list;

        for(int j=0; j<numBranches ;j++)   
        {
            if (j > i)
            {
                std::vector<std::map<std::string, std::string>> sequential_layer;
                std::map<std::string, std::string> conv_layer = {
                    {"type", "convolution"},
                    {"channel_size", std::to_string(numInChannels[i])},
                    {"kernel_size", "1"},
                    {"stride", "1"},
                    {"padding", "0"},
                    {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j) + ".0.weight"}
                };
                std::map<std::string, std::string> bn_layer = {
                    {"type", "batch_norm"},
                    {"eps", std::to_string(1e-5)},
                    {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j) + ".1"}
                };
                sequential_layer.push_back(conv_layer);
                sequential_layer.push_back(bn_layer);
                fuse_layer_module_list.push_back(sequential_layer);
            }
            else if (j == i)
            {
                std::vector<std::map<std::string, std::string>> sequential_layer;
                std::map<std::string, std::string> none = {
                    {"type", "none"}
                };
                sequential_layer.push_back(none);
                fuse_layer_module_list.push_back(sequential_layer);
            }
            else
            {
                std::vector<std::map<std::string, std::string>> sequential_layer;
                int sequential_layer_inlayer_iterator=0;
                for (int k=0; k<i-j; k++)
                {
                    if (k == i-j-1)
                    {
                        std::map<std::string, std::string> conv_layer = {
                            {"type", "convolution"},
                            {"channel_size", std::to_string(numInChannels[i])},
                            {"kernel_size", "3"},
                            {"stride", "2"},
                            {"padding", "1"},
                            {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j) + "." + std::to_string(sequential_layer_inlayer_iterator) + ".0.weight"}
                        };
                        std::map<std::string, std::string> bn_layer = {
                            {"type", "batch_norm"},
                            {"eps", std::to_string(1e-5)},
                            {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j)+ "."+std::to_string(sequential_layer_inlayer_iterator) + ".1"}
                        };
                        sequential_layer.push_back(conv_layer);
                        sequential_layer.push_back(bn_layer);
                        sequential_layer_inlayer_iterator+=1;
                    }
                    else
                    {
                        std::map<std::string, std::string> conv_layer = {
                            {"type", "convolution"},
                            {"channel_size", std::to_string(numInChannels[j])},
                            {"kernel_size", "3"},
                            {"stride", "2"},
                            {"padding", "1"},
                            {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j) + "." + std::to_string(sequential_layer_inlayer_iterator) + ".0.weight"}
                        };
                        std::map<std::string, std::string> bn_layer = {
                            {"type", "batch_norm"},
                            {"eps", std::to_string(1e-5)},
                            {"layer_name", lname + "." + std::to_string(i) + "." + std::to_string(j) + "." + std::to_string(sequential_layer_inlayer_iterator)+ ".1"}
                        };
                        std::map<std::string, std::string> relu_layer = {
                            {"type", "relu"}
                        };
                        sequential_layer.push_back(conv_layer);
                        sequential_layer.push_back(bn_layer);
                        sequential_layer.push_back(relu_layer);
                        sequential_layer_inlayer_iterator+=1;
                    }
                }
                fuse_layer_module_list.push_back(sequential_layer);
            }
        }

    fuse_layers_module_list.push_back(fuse_layer_module_list);
    }
    
    return fuse_layers_module_list;
}


nvinfer1::ITensor* HrnetBackbone::AddConfigLayer(nvinfer1::ITensor* input,  std::vector<std::map<std::string, std::string>> seqLayerInfo)
{
    for (int sequential_it = 0; sequential_it < seqLayerInfo.size(); sequential_it++)
    {
        std::map<std::string, std::string>& layer_info = seqLayerInfo[sequential_it];
        std::string layer_type = layer_info["type"];
    
        if(layer_type == "convolution"){

            int channel_size = std::stoi( layer_info["channel_size"] );
            int kernel_size  = std::stoi( layer_info["kernel_size"] );
            int padding      = std::stoi( layer_info["padding"] );
            int stride       = std::stoi( layer_info["stride"] );

            std::string layer_name = layer_info["layer_name"];
            nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

            nvinfer1::IConvolutionLayer *conv = mpNetwork->addConvolution(*input, channel_size, nvinfer1::DimsHW{kernel_size, kernel_size}, mWeightMap[layer_name], emptywts);
            assert(conv);
            conv->setStride(nvinfer1::DimsHW{stride, stride});
            conv->setPadding(nvinfer1::DimsHW{padding, padding});
            input = conv->getOutput(0);
        }
        else if(layer_type == "batch_norm"){
            float eps = std::stof(layer_info["eps"]);
            std::string layer_name = layer_info["layer_name"];
            nvinfer1::IScaleLayer* bn = addBatchNorm2d(mpNetwork, mWeightMap, *input, layer_name, eps);
            input = bn->getOutput(0);
        }
        else if(layer_type == "relu"){
            nvinfer1::IActivationLayer* relu = mpNetwork->addActivation(*input, nvinfer1::ActivationType::kRELU);
            assert(relu);
            input = relu->getOutput(0);
        }  
        else if(layer_type == "none"){
            continue;
        }  
    }

    return input;
}


std::vector<nvinfer1::ITensor *> HrnetBackbone::AddHighResolutionModule(std::vector<nvinfer1::ITensor*> inputs, std::string lname, int numBranches, 
                                                                        std::string blockType, std::vector<int> numBlocks, std::vector<int> numInChannels, 
                                                                        std::vector<int> numChannels, std::string fuseMethod, bool multiScaleOutput, int stride)
{        
    assert(numBranches == numBlocks.size());
    assert(numBranches == numChannels.size());
    assert(numBranches == numInChannels.size());

    for(int branch_it=0; branch_it< numBranches; branch_it++)
    {
        bool has_downsample = false;
        int block_expansion = blockType == "BOTTLENECK"? mBottleneckExpansion:mBasicExpansion;
        int block_channel_multiplication = numChannels[branch_it] * block_expansion;

        if (stride != 1 || numInChannels[branch_it] != block_channel_multiplication){
            has_downsample = true;
        }

        if(blockType == "BASIC")
        {
            inputs[branch_it]  = AddBasicBlock(inputs[branch_it], lname + ".branches." + std::to_string(branch_it) +  ".0"  , nvinfer1::DimsHW{stride, stride}, block_channel_multiplication, has_downsample);
            
            for(int i=1 ; i< numBlocks[branch_it] ;i++)
                inputs[branch_it]  = AddBasicBlock(inputs[branch_it], lname + ".branches." + std::to_string(branch_it) + "." + std::to_string(i), nvinfer1::DimsHW{stride, stride}, block_channel_multiplication, false);
        }
        else {
            std::cout<< "NOT IMPLEMENTED YET!" << std::endl;
        }
    }
    if (numBranches == 1)
        return std::vector<nvinfer1::ITensor *>{inputs[0]};

    int fuse_layer_count = multiScaleOutput ? numBranches : 1;
    auto fuse_layers = MakeFuseLayers (lname + ".fuse_layers",  numBranches, numInChannels, fuse_layer_count);

    std::vector<nvinfer1::ITensor *> x_fuse;
    nvinfer1::ITensor *y;
    for (int i=0; i<fuse_layer_count; i++)
    {
        y = inputs[0];

        if (i != 0){
            y = AddConfigLayer(y, fuse_layers[i][0]);
        }

        for (int j=1; j < numBranches; j++)
        {
            if (i == j)
            {
                nvinfer1::IElementWiseLayer *ew = mpNetwork->addElementWise(*y, *inputs[j], nvinfer1::ElementWiseOperation::kSUM);
                y = ew->getOutput(0);
            }
            else if(j > i)
            {
                auto dims = inputs[i]->getDimensions();


                nvinfer1::ITensor *tmp = AddConfigLayer(inputs[j], fuse_layers[i][j]);
                
                // nvinfer1::IResizeLayer* interpolate = mpNetwork->addResize(*tmp);
                // assert(interpolate);
                // interpolate->setOutputDimensions(nvinfer1::Dims3{tmp->getDimensions().d[0], height_output, width_output});
                // interpolate->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
                // interpolate->setAlignCorners(true);

                //auto interpolation_plugin = new Interpolation(tmp->getDimensions().d[0], height_output, width_output);
                //nvinfer1::ITensor *inputTensors[1]{tmp};

                //auto interpolate = mpNetwork->addPluginExt(inputTensors, 1, *interpolation_plugin);

                //interpolate->setName(Interpolation::GenerateLayerName());


                nvinfer1::IResizeLayer *interpolate = mpNetwork->addResize(*tmp);
                assert(interpolate);

                if (mpNetwork->hasImplicitBatchDimension()){
                    int channel = tmp->getDimensions().d[0];
                    int height_output = dims.d[1];
                    int width_output = dims.d[2];
                    interpolate->setOutputDimensions(nvinfer1::Dims{3, {channel, height_output, width_output}});
                }
                else{
                    int N = tmp->getDimensions().d[0];
                    int channel = tmp->getDimensions().d[1];
                    int height_output = dims.d[2];
                    int width_output = dims.d[3];
                    interpolate->setOutputDimensions(nvinfer1::Dims{4, {N, channel, height_output, width_output}});
                }
                interpolate->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
                interpolate->setAlignCorners(false);

                nvinfer1::IElementWiseLayer *ew = mpNetwork->addElementWise(*y, *interpolate->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

                y = ew->getOutput(0);
            }
            else
            {
                nvinfer1::ITensor *tmp = AddConfigLayer(inputs[j], fuse_layers[i][j]);
                nvinfer1::IElementWiseLayer *ew = mpNetwork->addElementWise(*y, *tmp, nvinfer1::ElementWiseOperation::kSUM);
                y = ew->getOutput(0);
            }
        }

        nvinfer1::IActivationLayer* relu_y = mpNetwork->addActivation(*y, nvinfer1::ActivationType::kRELU);
        assert(relu_y);

        x_fuse.push_back(relu_y->getOutput(0));
    }
    
    return x_fuse;
}


std::vector<nvinfer1::ITensor *> HrnetBackbone::MakeStage(std::vector<nvinfer1::ITensor*> inputs, std::string lname, std::vector<int> numInChannels, YAML::Node &stageConfig, bool multiScaleOutput)
{        
    int num_modules = stageConfig["NUM_MODULES"].as<int>();
    int num_branches = stageConfig["NUM_BRANCHES"].as<int>();
    std::vector<int> num_blocks = stageConfig["NUM_BLOCKS"].as<std::vector<int>>();
    std::vector<int> num_channels = stageConfig["NUM_CHANNELS"].as<std::vector<int>>();
    std::string block_type = stageConfig["BLOCK"].as<std::string>();
    std::string fuse_method = stageConfig["FUSE_METHOD"].as<std::string>();

    for (int i = 0; i < num_modules; i++)
    {
        bool reset_multi_scale_output;
        if ( !multiScaleOutput && i == num_modules- 1)
            reset_multi_scale_output = false;
        else
            reset_multi_scale_output = true;
    
        inputs = AddHighResolutionModule(inputs, lname + "." + std::to_string(i), num_branches, block_type, num_blocks, numInChannels, num_channels, fuse_method, reset_multi_scale_output, 1);
    }
    return inputs;
}


std::vector<nvinfer1::ITensor*> HrnetBackbone::Run(nvinfer1::ITensor* input)
{   
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv1 = mpNetwork->addConvolution(*input, 64, nvinfer1::DimsHW{3, 3}, mWeightMap["backbone.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(nvinfer1::DimsHW{2, 2});
    conv1->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(mpNetwork, mWeightMap, *conv1->getOutput(0),"backbone.bn1", 1e-5);

    nvinfer1::IActivationLayer* relu1 = mpNetwork->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IConvolutionLayer* conv2 = mpNetwork->addConvolution(*relu1->getOutput(0), 64, nvinfer1::DimsHW{3, 3}, mWeightMap["backbone.conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(nvinfer1::DimsHW{2, 2});
    conv2->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IScaleLayer* bn2 = addBatchNorm2d(mpNetwork, mWeightMap, *conv2->getOutput(0),"backbone.bn2", 1e-5);

    nvinfer1::IActivationLayer* relu2 = mpNetwork->addActivation(*bn2->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);

    YAML::Node stage_1_config = mConfig["MODEL"]["EXTRA"]["STAGE1"];
    int stage_1_num_channels = stage_1_config["NUM_CHANNELS"].as<std::vector<int>>()[0];
    int stage_1_num_blocks = stage_1_config["NUM_BLOCKS"].as<std::vector<int>>()[0];
    std::string stage_1_block_type = stage_1_config["BLOCK"].as<std::string>();

    nvinfer1::ITensor *layer1 = MakeLayer(*relu2->getOutput(0) , "backbone.layer1", stage_1_block_type, 64, stage_1_num_channels, stage_1_num_blocks);

    int stage_1__out_channel = stage_1_block_type == "BOTTLENECK" ? mBottleneckExpansion : mBasicExpansion;
    stage_1__out_channel *= stage_1_num_channels;
    std::vector<int> stage_1_out_channel_list={stage_1__out_channel};

    YAML::Node stage_2_config = mConfig["MODEL"]["EXTRA"]["STAGE2"];
    std::vector<int> stage_2_num_channels = stage_2_config["NUM_CHANNELS"].as<std::vector<int>>();
    std::string stage_2_block_type = stage_2_config["BLOCK"].as<std::string>();
    int block_expansion = stage_2_block_type == "BOTTLENECK" ? mBottleneckExpansion : mBasicExpansion;

    for (int i = 0; i < stage_2_num_channels.size(); i++)
    {
        stage_2_num_channels[i] *= block_expansion;
    }

    std::vector<nvinfer1::ITensor *> transition_1_input = {layer1};
    std::vector<nvinfer1::ITensor *> transition1 = MakeTransitionLayer(transition_1_input, "backbone.transition1", stage_1_out_channel_list, stage_2_num_channels);

    std::vector<nvinfer1::ITensor *> stage2_out = MakeStage(transition1, "backbone.stage2", stage_2_num_channels, stage_2_config, true);

    YAML::Node stage_3_config = mConfig["MODEL"]["EXTRA"]["STAGE3"];
    std::vector<int> stage_3_num_channels = stage_3_config["NUM_CHANNELS"].as<std::vector<int>>();
    std::string stage_3_block_type = stage_3_config["BLOCK"].as<std::string>();
    block_expansion = stage_3_block_type == "BOTTLENECK" ? mBottleneckExpansion : mBasicExpansion;

    for (int i = 0; i < stage_3_num_channels.size(); i++)
    {
        stage_3_num_channels[i] *= block_expansion;
    }

    std::vector<nvinfer1::ITensor *> transition2 = MakeTransitionLayer(stage2_out, "backbone.transition2", stage_2_num_channels, stage_3_num_channels);
    std::vector<nvinfer1::ITensor *> stage3_out = MakeStage(transition2, "backbone.stage3", stage_3_num_channels, stage_3_config, true);

    YAML::Node stage_4_config = mConfig["MODEL"]["EXTRA"]["STAGE4"];
    std::vector<int> stage_4_num_channels = stage_4_config["NUM_CHANNELS"].as<std::vector<int>>();
    std::string stage_4_block_type = stage_4_config["BLOCK"].as<std::string>();
    block_expansion = stage_4_block_type == "BOTTLENECK" ? mBottleneckExpansion : mBasicExpansion;

    for (int i = 0; i < stage_4_num_channels.size(); i++)
    {
        stage_4_num_channels[i] *= block_expansion;
    }

    std::vector<nvinfer1::ITensor *> transition3 = MakeTransitionLayer(stage3_out, "backbone.transition3", stage_3_num_channels, stage_4_num_channels);
    std::vector<nvinfer1::ITensor *> stage4_out = MakeStage(transition3, "backbone.stage4", stage_4_num_channels, stage_4_config, true);

    return stage4_out;

}       
