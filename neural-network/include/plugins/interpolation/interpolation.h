
#ifndef INTERPOLATION_PLUGIN_H
#define INTERPOLATION_PLUGIN_H

#include "NvInfer.h"
#include "logs.h"


class Interpolation : public nvinfer1::IPluginExt {
public:
    explicit Interpolation(int channel, int height, int width);

    Interpolation(const void *data, size_t length);

    int getNbOutputs() const override { return 1; }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override{
        return type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR;
    }

    void configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs, const nvinfer1::Dims *outputDims, int nbOutputs,
                            nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override;

    int initialize() override;

    void terminate() override{};

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    cudaError_t DoInterpolation(int outputChannel, int outputHeight, int outputWidth, int inputChannel, int inputHeight, int inputWidth,
                                const float* input, float* output, cudaStream_t* stream);

    static const char* GenerateLayerName();

private:
    int mOutputChannel;
    int mOutputHeight;
    int mOutputWidth;
    int mInputChannel;
    int mInputHeight;
    int mInputWidth;
    static int s_counter;
};

#endif

