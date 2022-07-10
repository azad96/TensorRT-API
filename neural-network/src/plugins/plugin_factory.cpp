#include "plugin_factory.h"
#include "interpolation.h"


nvinfer1::IPluginExt* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    nvinfer1::IPluginExt *plugin = nullptr;
    if (strstr(layerName, "interpolate") != NULL) {
        plugin = new Interpolation(serialData, serialLength);
    }
    return plugin;
}