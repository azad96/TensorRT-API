#include "resnet34.h"

using namespace Project;

int main(int argc, char** argv){
    int INPUT_C = 3;
    int INPUT_H = 224;
    int INPUT_W = 224;
    int OUTPUT_COUNT = 1000;
    unsigned int batch_size = 1;
    std::string weight_path = "../resources/resnet34.wts";
    size_t inputSize = INPUT_C * INPUT_H * INPUT_W;
    float* data = new float[batch_size * inputSize];
    std::fill(data, data+inputSize, 1.0);

    Resnet34 *network = new Resnet34(weight_path, INPUT_C, INPUT_H, INPUT_W, OUTPUT_COUNT, batch_size);
    
    auto start = std::chrono::system_clock::now();
    network->Init();
    network->DoInference(data);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    for (int i = 0; i < OUTPUT_COUNT; i++)
        std::cout << network->GetOutputPointerCPU()[i] << " ";
    std::cout << std::endl;

    delete network;
    return 0;
}
