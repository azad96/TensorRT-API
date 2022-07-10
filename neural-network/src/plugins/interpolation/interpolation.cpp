#include "interpolation.h"
#include <cassert>


int Interpolation::s_counter = 1;


Interpolation::Interpolation(int channel, int height, int width):
    mOutputChannel(channel), mOutputHeight(height), mOutputWidth(width) {}


Interpolation::Interpolation(const void *data, size_t length)
{
    assert(length == 6*sizeof(int));

    auto bufferi = (int*)(data);
    mOutputChannel = bufferi[0];
    mOutputHeight = bufferi[1];
    mOutputWidth = bufferi[2];
    mInputChannel = bufferi[3];
    mInputHeight = bufferi[4];
    mInputWidth = bufferi[5];
}


nvinfer1::Dims Interpolation::getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
    mInputChannel = inputs[0].d[0];
    mInputHeight = inputs[0].d[1];
    mInputWidth = inputs[0].d[2];

    assert(mInputChannel == mOutputChannel);

    nvinfer1::Dims3 res(mOutputChannel, mOutputHeight, mOutputWidth);

    return res;
}


void Interpolation::configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs, const nvinfer1::Dims *outputDims, int nbOutputs, 
                                    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs > 0);
}


int Interpolation::initialize()
{
    return 0;
}


size_t Interpolation::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}


int Interpolation::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    DoInterpolation(mOutputChannel, mOutputHeight, mOutputWidth, mInputChannel, mInputHeight, mInputWidth, (float *)inputs[0], (float *)outputs[0], &stream);
}


size_t Interpolation::getSerializationSize()
{
    return (6 * sizeof(int));
}


void Interpolation::serialize(void *buffer)
{
    auto bufferi = (int*)(buffer);

    bufferi[0] = mOutputChannel;
    bufferi[1] = mOutputHeight;
    bufferi[2] = mOutputWidth;
    bufferi[3] = mInputChannel;
    bufferi[4] = mInputHeight;
    bufferi[5] = mInputWidth;
}


const char* Interpolation::GenerateLayerName()
{
    static char s_pluginName[16];
    sprintf(s_pluginName, "%s%d", "interpolate", s_counter++);
    return s_pluginName;
}