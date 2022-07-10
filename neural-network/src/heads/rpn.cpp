//
// Created by botan on 14.12.2020.
//

#include "rpn.h"
#include "common.h"

Rpn::Rpn(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, int number_of_anchors):
    mpNetwork(network), mWeightMap(weightMap), mNumberOfAnchors(number_of_anchors)
{
    mExplicitBatchOffset = mpNetwork->hasImplicitBatchDimension() ? 0 : 1;
}


std::vector<nvinfer1::ITensor*> Rpn::SingleRun(nvinfer1::ITensor* input)
{
    /*
     * We have multiple feature levels. At each feature level, we will regress bbox and predict class for possible locations
     * For each feature level SingleRun will be called and class result and std vectors will be returned in a std::vector.
     */
    std::vector<nvinfer1::ITensor*> rpn_outs;
    nvinfer1::IConvolutionLayer* rpn_conv = mpNetwork->addConvolution(*input,256,nvinfer1::DimsHW{3, 3},
                                                                            mWeightMap["rpn_head.rpn_conv.weight"],
                                                                            mWeightMap["rpn_head.rpn_conv.bias"]);
    assert(rpn_conv);
    rpn_conv->setPadding(nvinfer1::DimsHW{1, 1});

    nvinfer1::IActivationLayer* rpn_relu = mpNetwork->addActivation(*rpn_conv->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(rpn_relu);


    nvinfer1::IConvolutionLayer* rpn_cls = mpNetwork->addConvolution(*rpn_relu->getOutput(0), mNumberOfAnchors, nvinfer1::DimsHW{1, 1},
                                                                      mWeightMap["rpn_head.rpn_cls.weight"],
                                                                      mWeightMap["rpn_head.rpn_cls.bias"]);
    assert(rpn_cls);

    nvinfer1::ITensor* cls_result = rpn_cls->getOutput(0);

    rpn_outs.push_back(cls_result);

    nvinfer1::IConvolutionLayer* rpn_bbox = mpNetwork->addConvolution(*rpn_relu->getOutput(0), mNumberOfAnchors*4 ,nvinfer1::DimsHW{1, 1},
                                                                     mWeightMap["rpn_head.rpn_reg.weight"],
                                                                     mWeightMap["rpn_head.rpn_reg.bias"]);
    assert(rpn_bbox);

    nvinfer1::ITensor* bbox_result = rpn_bbox->getOutput(0);

    rpn_outs.push_back(bbox_result);

    return rpn_outs;

}

std::vector<nvinfer1::ITensor*> Rpn::Run(std::vector<nvinfer1::ITensor *> &inputs, std::vector<nvinfer1::ITensor *> &multi_level_anchors)

{
    int input_nums = inputs.size(); // input nums corresponds to feature level number.

    std::vector<std::vector<nvinfer1::ITensor*>> rpn_outs(2);  // first index corresponds to feature level object probabilities
                                                                  // second index corresponds to feature level bbox prediction(deltas)

    // call SingleRun to calculate object probabilities and bboxs at each feature level.
    for(int i=0; i<input_nums; i++)
    {
        std::vector<nvinfer1::ITensor*> current_rpn_outs = SingleRun(inputs[i]);
        rpn_outs[0].push_back(current_rpn_outs[0]);  // object probabilities
        rpn_outs[1].push_back(current_rpn_outs[1]); // bbox predictions(deltas)
    }

    assert(rpn_outs[0].size() == rpn_outs[1].size()); // number of feature levels at bbox and scores output should be same

    std::vector<int> level_ids;
    std::vector<nvinfer1::ITensor*> multi_level_scores;
    std::vector<nvinfer1::ITensor*> multi_level_bboxs; // deltas
    std::vector<nvinfer1::ITensor*> multi_level_valid_anchors;

    // loop over feature levels
    for(int i=0; i<multi_level_anchors.size();i++)
    {
        nvinfer1::IShuffleLayer* resize_score_layer = mpNetwork->addShuffle(*rpn_outs[0][i]);
        assert(resize_score_layer);
        if (mpNetwork->hasImplicitBatchDimension()){
            resize_score_layer->setFirstTranspose(nvinfer1::Permutation{1, 2, 0});
            resize_score_layer->setReshapeDimensions(nvinfer1::Dims{1, {-1}});
        }
        else{
            resize_score_layer->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
            resize_score_layer->setReshapeDimensions(nvinfer1::Dims{2, {1, -1}});
        }
        nvinfer1::ITensor* scores_vector = resize_score_layer->getOutput(0);

        nvinfer1::IActivationLayer* sigmoid_layer = mpNetwork->addActivation(*scores_vector, nvinfer1::ActivationType::kSIGMOID);
        assert(scores_vector);
        nvinfer1::ITensor* cls_probability = sigmoid_layer->getOutput(0);

        nvinfer1::ITensor* rpn_bbox_pred = rpn_outs[1][i]; // deltas
        nvinfer1::IShuffleLayer* resize_bbox_layer = mpNetwork->addShuffle(*rpn_bbox_pred);
        assert(resize_bbox_layer);

        int cls_probability_size = 1;
        if (mpNetwork->hasImplicitBatchDimension()){
            resize_bbox_layer->setFirstTranspose(nvinfer1::Permutation{1, 2, 0});
            resize_bbox_layer->setReshapeDimensions(nvinfer1::Dims{2, {-1, 4}});
            cls_probability_size = cls_probability->getDimensions().d[0];
        }
        else{
            resize_bbox_layer->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
            resize_bbox_layer->setReshapeDimensions(nvinfer1::Dims{3, {1, -1, 4}});
            cls_probability_size = cls_probability->getDimensions().d[1];
        }
        nvinfer1::ITensor* bbox_vector = resize_bbox_layer->getOutput(0);
        nvinfer1::ITensor* anchors_of_current_level = multi_level_anchors[i];

        if(mNmsPre > 0 && cls_probability_size > mNmsPre)
        {
            // find indices of 1000 predictions with highest scores
            // reduceAxes 1 if implicit batch size is used, otherwise 2
            auto top_k_scores_layer = mpNetwork->addTopK(*cls_probability, nvinfer1::TopKOperation::kMAX, mNmsPre, 1 + mExplicitBatchOffset);
            assert(top_k_scores_layer);

            // axis 0 if implicit, otherwise 1
            auto gatherLayer_bbox = mpNetwork->addGather(*bbox_vector, *top_k_scores_layer->getOutput(1), mExplicitBatchOffset);
            assert(gatherLayer_bbox);
            gatherLayer_bbox->setNbElementWiseDims(mExplicitBatchOffset); 

            // axis 0 if implicit, otherwise 1
            auto gatherLayer_anchor = mpNetwork->addGather(*anchors_of_current_level, *top_k_scores_layer->getOutput(1), mExplicitBatchOffset);
            assert(gatherLayer_anchor);
            gatherLayer_anchor->setNbElementWiseDims(mExplicitBatchOffset); 

            nvinfer1::ITensor* proposed_bboxes = gatherLayer_bbox->getOutput(0);
            nvinfer1::ITensor* proposed_scores = top_k_scores_layer->getOutput(0);
            nvinfer1::ITensor* proposed_anchors = gatherLayer_anchor->getOutput(0);

            multi_level_scores.push_back(proposed_scores);
            multi_level_bboxs.push_back(proposed_bboxes);
            multi_level_valid_anchors.push_back(proposed_anchors);
            level_ids.push_back(i);
        }
        else{
            nvinfer1::ITensor* proposed_bboxes = bbox_vector;
            nvinfer1::ITensor* proposed_scores = cls_probability;
            nvinfer1::ITensor* proposed_anchors = anchors_of_current_level;

            multi_level_scores.push_back(proposed_scores);
            multi_level_bboxs.push_back(proposed_bboxes);
            multi_level_valid_anchors.push_back(proposed_anchors);
            level_ids.push_back(i);
        }
    }

    nvinfer1::ITensor* final_scores_list[]  = {multi_level_scores[0], multi_level_scores[1], multi_level_scores[2], multi_level_scores[3], multi_level_scores[4]};
    nvinfer1::ITensor* final_bboxs_list[]   = {multi_level_bboxs[0], multi_level_bboxs[1], multi_level_bboxs[2], multi_level_bboxs[3], multi_level_bboxs[4]};
    nvinfer1::ITensor* final_anchors_list[] = {multi_level_valid_anchors[0], multi_level_valid_anchors[1], multi_level_valid_anchors[2], multi_level_valid_anchors[3], multi_level_valid_anchors[4]};

    nvinfer1::IConcatenationLayer *concat_scores_layer = mpNetwork->addConcatenation(final_scores_list, multi_level_anchors.size());
    assert(concat_scores_layer);
    concat_scores_layer->setAxis(mExplicitBatchOffset); // axis 0 if implicit batch size is used, otherwise 1
    nvinfer1::ITensor* final_scores = concat_scores_layer->getOutput(0);

    nvinfer1::IConcatenationLayer *concat_bboxs_layer = mpNetwork->addConcatenation(final_bboxs_list, multi_level_anchors.size());
    assert(concat_bboxs_layer);
    concat_bboxs_layer->setAxis(mExplicitBatchOffset); // axis 0 if implicit batch size is used, otherwise 1
    nvinfer1::ITensor* final_deltas = concat_bboxs_layer->getOutput(0);

    nvinfer1::IConcatenationLayer *concat_anchors_layer = mpNetwork->addConcatenation(final_anchors_list, multi_level_anchors.size());
    assert(concat_anchors_layer);
    concat_anchors_layer->setAxis(mExplicitBatchOffset); // axis 0 if implicit batch size is used, otherwise 1
    nvinfer1::ITensor* final_rois = concat_anchors_layer->getOutput(0);

    // output: scores, deltas, rois
    std::vector<nvinfer1::ITensor*> rpn_final_result = {final_scores, final_deltas, final_rois};
    return rpn_final_result;
}