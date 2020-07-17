#include "MGPluginFactory.h"
#include "ClampingPlugin.h"
#include <string.h>

using namespace nvinfer1;
using nvinfer1::PluginFactoryMG;

IPluginExt* PluginFactoryMG::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPluginExt *plugin = nullptr;
    if (strstr(layerName, "cl") != NULL) {
        plugin = new ClampingPlugin(serialData, serialLength);
    }

    return plugin;
}