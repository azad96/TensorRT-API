#ifndef PLUGINFACTORY_H
#define PLUGINFACTORY_H
#include <NvInfer.h>

namespace nvinfer1 {
    class PluginFactoryMG : public IPluginFactory {
    public:
        IPluginExt* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
    };
}

#endif // PLUGINFACTORY_H
