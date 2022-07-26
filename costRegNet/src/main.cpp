#include "cost_reg_net.h"
#include <chrono>
#include <iomanip>


// stuff we know about the network
static const std::vector<const char*> INPUT_NAMES {"volume_variance"}; // 1, 32, 48, 120, 160
static const std::vector<const char*> OUTPUT_NAMES {"cost_reg"}; // 1, 1, 48, 120, 160


int main(int argc, char** argv){
    unsigned int batchSize = 1;
    std::string weightPath = "../weights/cost_reg_net.wts";

    int dim1 = 32;
    int dim2 = 48;
    int dim3 = 120;
    int dim4 = 160;

    std::vector<std::vector<int>> inputDims {
        {dim1, dim2, dim3, dim4} 
    }; 

    size_t inputSize = dim1 * dim2 * dim3 * dim4;
    float* data = new float[batchSize * inputSize];

    std::ifstream file("../volume_variance.txt");
    std::string num; 
    int i = 0;

    while (std::getline(file, num)){
        data[i++] = std::stof(num);
    }

    // for (int i = 0; i < inputSize; i += inputSize/10)
    //     std::cout << data[i] << std::endl;

    float* inputs[1] {data};

    TensorRT *network = new CostRegNet(INPUT_NAMES, OUTPUT_NAMES, inputDims, batchSize, false);
    network->PrepareInference(weightPath);
    network->Init();

    for (int k=0; k<10; k++){
        auto start = std::chrono::system_clock::now();
        network->DoInference(1, inputs);
        auto end = std::chrono::system_clock::now();
        std::cout << "Inference took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    }

    // std::cout << std::fixed << std::setprecision(6) << std::endl;

    // auto outputDims = network->GetTensorDims(OUTPUT_NAMES[0]);
    // int outSize = 1;
    // std::cout << "output shape: ";
    // for (int j = 0; j < outputDims.nbDims; j++){
    //     outSize *= outputDims.d[j];
    //     std::cout << outputDims.d[j] << " ";
    // }
    // std::cout << std::endl;
        
    // for (int j = 0; j < outSize; j += outSize/10) {
    //     std::cout << network->GetOutputPointersCPU(0)[j] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;

    // CUDA EVENTS
    delete network;
    return 0;
}
