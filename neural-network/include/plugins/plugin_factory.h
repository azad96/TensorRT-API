#ifndef INTERPOLATION_PLUGIN_FACTORY_H
#define INTERPOLATION_PLUGIN_FACTORY_H
#include <NvInfer.h>

class PluginFactory : public nvinfer1::IPluginFactory {
public:
    nvinfer1::IPluginExt* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
};

#endif 
