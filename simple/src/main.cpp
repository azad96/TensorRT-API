#include "simple.h"
#include <chrono>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv){
    unsigned int batchSize = 1;
    std::string weightPath = "../weights/simple.wts";

    int channel = 3;
    int height = 4;
    int width = 4;

    std::vector<std::vector<int>> inputDims {
        {channel, height, width} 
    }; 

    size_t inputSize = channel * height * width;
    float* data = new float[batchSize * inputSize];

    for (int i=0; i<inputSize; i++)
        data[i] = i;

    float* inputs[1] {data};

    TensorRT *network = new SIMPLE(INPUT_NAMES, OUTPUT_NAMES, PROJECT_NAME, 
                                inputDims, batchSize, true);
    network->PrepareInference(weightPath);

    auto start = std::chrono::system_clock::now();
    network->Init();
    network->DoInference(1, inputs);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    auto outputDims = network->GetTensorDims(OUTPUT_NAMES[0]);
    int outSize = 1;
        for (int j = 0; j < outputDims.nbDims; j++)
            outSize *= outputDims.d[j];

    for (int i = 0; i < outSize; i++) {
        std::cout << network->GetOutputPointersCPU(0)[i] << " ";
    }
    std::cout << std::endl;

    delete network;
    return 0;
}
