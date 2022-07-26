#include "featurenet.h"
#include <chrono>
#include <iomanip>

// stuff we know about the network
static const std::vector<const char*> INPUT_NAMES {"data"};
static const std::vector<const char*> OUTPUT_NAMES {"stage1", "stage2", "stage3"};


int main(int argc, char** argv){
    unsigned int batchSize = 1;
    std::string weightPath = "../weights/featurenet.wts";

    int view = 2;
    int channel = 3;
    int height = 480;
    int width = 640;

    std::vector<std::vector<int>> inputDims {
        {view, channel, height, width} 
    }; 

    size_t inputSize = view * channel * height * width;
    float* data = new float[batchSize * inputSize];

    std::ifstream file("../input.txt");
    std::string num; 
    int i = 0;

    while (std::getline(file, num)){
        data[i++] = std::stof(num);
    }

    float* inputs[1] {data};

    TensorRT *network = new FeatureNet(INPUT_NAMES, OUTPUT_NAMES, inputDims, batchSize, false);
    network->PrepareInference(weightPath);

    network->Init();

    for (int k=0; k<10; k++){
        auto start = std::chrono::system_clock::now();
        network->DoInference(1, inputs);
        auto end = std::chrono::system_clock::now();
        std::cout << "Inference took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    }

    // std::cout << std::fixed << std::setprecision(6) << std::endl;

    // for (int i = 0; i < 3; i++){
    //     auto outputDims = network->GetTensorDims(OUTPUT_NAMES[i]);
    //     int outSize = 1;
    //     std::cout << "stage" << i+1 << " shape: ";
    //     for (int j = 0; j < outputDims.nbDims; j++){
    //         outSize *= outputDims.d[j];
    //         std::cout << outputDims.d[j] << " ";
    //     }
    //     std::cout << std::endl;
            
    //     for (int j = 0; j < outSize; j += outSize/10) {
    //         std::cout << network->GetOutputPointersCPU(i)[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
    // CUDA EVENTS
    delete network;
    return 0;
}
