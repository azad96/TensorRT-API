#ifndef MULTIGRAIN_h
#define MULTIGRAIN_h

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <chrono>
#include <opencv2/opencv.hpp>

#define LOG_TRT "[TRT] "

// stuff we know about the network and the input/output blobs
static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static const char* PROJECT = "MultiGrain";

static Logger gLogger;

class MultiGrain {
public:
    explicit MultiGrain(std::string &weight_path, int in_channel, int in_width, int in_height, int output_count, unsigned int batch_size);
    ~MultiGrain();
    bool Init();
    bool DoInference(float* input);
    float *GetInputPointerGPU() { return m_pNetInputGPU; }
    float *GetOutputPointerGPU() { return m_pNetOutputGPU; }
    float *GetOutputPointerCPU() { return m_pNetOutputCPU; }
    const nvinfer1::Dims GetTensorDims(const char *name);
private:
    nvinfer1::ICudaEngine* CreateEngine(std::map<std::string, nvinfer1::Weights> weightMap, unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::DataType dt);
    bool AllocateMemory();
    nvinfer1::IExecutionContext *m_context = nullptr;
    nvinfer1::ICudaEngine *m_engine = nullptr;
    nvinfer1::IRuntime *m_runtime = nullptr;
    cudaStream_t m_stream = nullptr;
    int m_iChannel = 3;
    int m_iHeight = 224;
    int m_iWidth = 224;
    int m_iNumberOfClassess = 0;
    size_t m_networkOutputSize = 0;
    size_t m_networkInputSize = 0;
    unsigned int m_uiNetworkBatchSize = 0;
    bool errorOccurred = false;
    bool m_bInitialized = false;
    float *m_pNetInputGPU = nullptr;
    float *m_pNetOutputGPU = nullptr;
    float *m_pNetOutputCPU = nullptr;
};
#endif // MULTIGRAIN_h

