#ifndef CLAMPINGPLUGIN_H
#define CLAMPINGPLUGIN_H

#include "NvInfer.h"

namespace nvinfer1 {
    class ClampingPlugin : public IPluginExt {
    public:
        explicit ClampingPlugin();

        ~ClampingPlugin();

        ClampingPlugin(const void *data, size_t length);

        int getNbOutputs() const override {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        void
        configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                            PluginFormat format, int maxBatchSize) override;

        int initialize() override;

        void terminate() override {};

        size_t getWorkspaceSize(int maxBatchSize) const override;

        int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream) override;

        size_t getSerializationSize() override;

        void serialize(void *buffer) override;

    private :

        int m_iNumberOfElements;

    };
}

#endif // CLAMPINGPLUGIN_H
