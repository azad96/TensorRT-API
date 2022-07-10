//
// Created by azad on 20.04.2021.
//

#include "mask_head.h"
#include "common.h"
#include <numeric>

MaskHead::MaskHead(nvinfer1::INetworkDefinition* &network, std::map<std::string, nvinfer1::Weights> &weightMap)
{
    mpNetwork = network;
    mWeightMap = weightMap;
}


// nvinfer1::ITensor* MaskHead::DoPasteMask(nvinfer1::ITensor* &masks, nvinfer1::ITensor* &bboxes, int img_h, int img_w)
// {
//     int x0_int = 0, y0_int = 0;
//     int x1_int = img_w, y1_int = img_h;

//     int N = masks->getDimensions().d[0];

//     auto x0 = mpNetwork->addSlice(*bboxes, 
//                                 nvinfer1::Dims{2, {0, 0}}, 
//                                 nvinfer1::Dims{2, {bboxes->getDimensions().d[0], 1}}, 
//                                 nvinfer1::Dims{2, {1, 0}});

//     auto y0 = mpNetwork->addSlice(*bboxes, 
//                                 nvinfer1::Dims{2, {0, 1}}, 
//                                 nvinfer1::Dims{2, {bboxes->getDimensions().d[0], 1}}, 
//                                 nvinfer1::Dims{2, {1, 0}});

//     auto x1 = mpNetwork->addSlice(*bboxes, 
//                                 nvinfer1::Dims{2, {0, 2}}, 
//                                 nvinfer1::Dims{2, {bboxes->getDimensions().d[0], 1}}, 
//                                 nvinfer1::Dims{2, {1, 0}});

//     auto y1 = mpNetwork->addSlice(*bboxes, 
//                                 nvinfer1::Dims{2, {0, 3}}, 
//                                 nvinfer1::Dims{2, {bboxes->getDimensions().d[0], 1}}, 
//                                 nvinfer1::Dims{2, {1, 0}});

//     assert(x0); assert(y0); assert(x1); assert(y1);

//     int img_y_size = y1_int - y0_int;
//     float* img_y_weights = new float[img_y_size];
//     std::iota(img_y_weights, img_y_weights + img_y_size, 0.5);
//     auto img_y = mpNetwork->addConstant(nvinfer1::Dims{2, {1, img_y_size}}, 
//                                     nvinfer1::Weights{nvinfer1::DataType::kFLOAT, img_y_weights, img_y_size});
//     assert(img_y);
    
//     // img_y - y0
//     auto y_nominator = mpNetwork->addElementWise(*img_y->getOutput(0), *y0->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(y_nominator);

//     // y1 - y0
//     auto y_denominator = mpNetwork->addElementWise(*y1->getOutput(0), *y0->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(y_denominator);

//     // (img_y - y0) / (y1 - y0)
//     auto y_division = mpNetwork->addElementWise(*y_nominator->getOutput(0), *y_denominator->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
//     assert(y_division);

//     // (img_y - y0) / (y1 - y0) * 2
//     const float multiplier_value = 2.0f;
//     auto multiplier = mpNetwork->addConstant(nvinfer1::Dims{2, {1, 1}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &multiplier_value, 1});
//     assert(multiplier);
//     auto y_division2 = mpNetwork->addElementWise(*y_division->getOutput(0), *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
//     assert(y_division2);

//     // (img_y - y0) / (y1 - y0) * 2 - 1
//     const float substract_value = 1.0f;
//     auto subtract = mpNetwork->addConstant(nvinfer1::Dims{2, {1, 1}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &substract_value, 1});
//     assert(subtract);
//     auto img_y_output_layer = mpNetwork->addElementWise(*y_division2->getOutput(0), *subtract->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(img_y_output_layer);

//     int img_x_size = x1_int - x0_int;
//     float* img_x_weights = new float[img_x_size];
//     std::iota(img_x_weights, img_x_weights + img_x_size, 0.5);
//     auto img_x = mpNetwork->addConstant(nvinfer1::Dims{2, {1, img_x_size}}, 
//                                     nvinfer1::Weights{nvinfer1::DataType::kFLOAT, img_x_weights, img_x_size});                                    
//     assert(img_x);

//     // img_x - x0
//     auto x_nominator = mpNetwork->addElementWise(*img_x->getOutput(0), *x0->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(x_nominator);

//     // x1 - x0
//     auto x_denominator = mpNetwork->addElementWise(*x1->getOutput(0), *x0->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(x_denominator);

//     // (img_x - x0) / (x1 - x0)
//     auto x_division = mpNetwork->addElementWise(*x_nominator->getOutput(0), *x_denominator->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
//     assert(x_division);

//     // (img_x - x0) / (x1 - x0) * 2
//     auto x_division2 = mpNetwork->addElementWise(*x_division->getOutput(0), *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
//     assert(x_division2);

//     // (img_x - x0) / (x1 - x0) * 2 - 1
//     auto img_x_output_layer = mpNetwork->addElementWise(*x_division2->getOutput(0), *subtract->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
//     assert(img_x_output_layer);
    
//     /*
//     ** Infinity values in img_x_output and img_y_output should be converted to 0. probably with a plugin layer 
//     */

//     // expand img_y from {d0, 810} to {d0, 810, 1080, 1}
//     auto img_y_dims = img_y_output_layer->getOutput(0)->getDimensions();
//     auto reshaped_img_y = mpNetwork->addShuffle(*img_y_output_layer->getOutput(0));
//     assert(reshaped_img_y);
//     reshaped_img_y->setReshapeDimensions(nvinfer1::Dims{4, {img_y_dims.d[0], img_y_dims.d[1], 1, 1}});

//     nvinfer1::ITensor* img_y_concat_list[img_w];
//     for (int i = 0; i < img_w; i++)
//         img_y_concat_list[i] = reshaped_img_y->getOutput(0);
//     auto img_y_concat = mpNetwork->addConcatenation(img_y_concat_list, img_w);
//     assert(img_y_concat);
//     img_y_concat->setAxis(2);

//     // expand img_x from {d0, 1080} to {d0, 810, 1080, 1}
//     auto img_x_dims = img_x_output_layer->getOutput(0)->getDimensions();
//     auto reshaped_img_x = mpNetwork->addShuffle(*img_x_output_layer->getOutput(0));
//     assert(reshaped_img_x);
//     reshaped_img_x->setReshapeDimensions(nvinfer1::Dims{4, {img_x_dims.d[0], 1, img_x_dims.d[1], 1}});

//     nvinfer1::ITensor* img_x_concat_list[img_h];
//     for (int i = 0; i < img_h; i++)
//         img_x_concat_list[i] = reshaped_img_x->getOutput(0);
//     auto img_x_concat = mpNetwork->addConcatenation(img_x_concat_list, img_h);
//     assert(img_x_concat);
//     img_x_concat->setAxis(1);

//     // concat img_x_concat and img_y_concat
//     nvinfer1::ITensor* grid_list[2] = {img_x_concat->getOutput(0), img_y_concat->getOutput(0)};
//     auto grid = mpNetwork->addConcatenation(grid_list, 2);
//     assert(grid);
//     grid->setAxis(3);

//     return grid->getOutput(0);
// }


// nvinfer1::ITensor* MaskHead::GetSegMasks(nvinfer1::ITensor* &masks, nvinfer1::ITensor* &bboxes, std::vector<int> ori_shape)
// {
//     // rescale = false is not implemented
//     assert(mRescale);
//     int img_h = ori_shape[1];
//     int img_w = ori_shape[2];

//     // if scale factor is not 1.0 bboxes/scale_factor should be added

//     // int N = mask_pred->getDimensions().d[0];

//     // return DoPasteMask(mask_pred, det_bboxes, img_h, img_w);
// }


nvinfer1::ITensor* MaskHead::Run(nvinfer1::ITensor* mask_feats, nvinfer1::ITensor* &last_feat, int head_id, int class_number)
{
    nvinfer1::ITensor* x = mask_feats;

    if (last_feat){
        auto conv_res = mpNetwork->addConvolution(*last_feat, 256, nvinfer1::DimsHW{1, 1},
                                                mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".conv_res.conv.weight"],
                                                mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".conv_res.conv.bias"]);
        assert(conv_res);

        auto relu_res = mpNetwork->addActivation(*conv_res->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu_res);

        last_feat = relu_res->getOutput(0);

        auto ew1 = mpNetwork->addElementWise(*x, *last_feat, nvinfer1::ElementWiseOperation::kSUM);
        assert(ew1);
        x = ew1->getOutput(0);
    }

    for (int i = 0; i < mConvLayerCount; i++){
        auto weight = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".convs." + std::to_string(i) + ".conv.weight"];
        auto bias = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".convs." + std::to_string(i) + ".conv.bias"];

        auto conv_i = mpNetwork->addConvolution(*x, 256, nvinfer1::DimsHW{3, 3}, weight, bias);
        assert(conv_i);
        conv_i->setStride(nvinfer1::DimsHW{1, 1});
        conv_i->setPadding(nvinfer1::DimsHW{1, 1});

        auto relu_i = mpNetwork->addActivation(*conv_i->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu_i);

        x = relu_i->getOutput(0);
    }
    
    last_feat = x;
    
    auto weight = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".upsample.weight"];
    auto bias = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".upsample.bias"];
    nvinfer1::IDeconvolutionLayer* upsample = mpNetwork->addDeconvolution(*x, 256, nvinfer1::DimsHW{2, 2}, weight, bias);
    assert(upsample);
    upsample->setStride(nvinfer1::DimsHW{2, 2});

    auto relu_upsample = mpNetwork->addActivation(*upsample->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu_upsample);    

    x = relu_upsample->getOutput(0);

    weight = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".conv_logits.weight"];
    bias = mWeightMap["roi_head.mask_head." + std::to_string(head_id) + ".conv_logits.bias"];
    auto conv_logits = mpNetwork->addConvolution(*x, class_number, nvinfer1::DimsHW{1, 1}, weight, bias);
    assert(conv_logits);

    auto mask_pred = conv_logits->getOutput(0);

    return mask_pred;
}