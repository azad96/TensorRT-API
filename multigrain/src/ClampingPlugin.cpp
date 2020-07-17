#include "ClampingPlugin.h"
#include "ClampingPluginKernel.h"
#include <cuda_runtime.h>
#include <cassert>
#include <string.h>


namespace nvinfer1
{
    ClampingPlugin::ClampingPlugin(){}

    ClampingPlugin::~ClampingPlugin(){}

    ClampingPlugin::ClampingPlugin(const void *data, size_t length)
    {
        assert(length == sizeof(int));

        auto bufferi = (int*)(data);
        m_iNumberOfElements = bufferi[0];
    }

    Dims ClampingPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
    {
        assert(nbInputDims>0);
        nvinfer1::Dims res = inputs[0];
        m_iNumberOfElements = 1;
        for(int i = 0; i< inputs[0].nbDims; ++i)
        {
            m_iNumberOfElements *= inputs[0].d[i];
        }
        return res;
    }

    void ClampingPlugin::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                    PluginFormat format, int maxBatchSize)
    {
        assert(nbInputs > 0);
    }

    int ClampingPlugin::initialize()
    {
        return 0;
    }


    size_t ClampingPlugin::getWorkspaceSize(int maxBatchSize) const
    {
        return 0;
    }

    int ClampingPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
            cudaStream_t stream)
    {
        clamping(m_iNumberOfElements, (float*)inputs[0], (float*)outputs[0], &stream);
    }

    size_t ClampingPlugin::getSerializationSize()
    {
        return (1* sizeof(int));
    }

    void ClampingPlugin::serialize(void *buffer)
    {
        auto bufferi = (int*)(buffer);
        bufferi[0] = m_iNumberOfElements;
    }
}