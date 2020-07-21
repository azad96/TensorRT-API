#include "multigrain.h"

using namespace Project;

const float means[3] = {0.485, 0.456, 0.406};
const float stds[3]  = {0.229, 0.224, 0.225};

void process_image(float* data, std::string image_name, int INPUT_H, int INPUT_W){
    cv::Mat img = cv::imread(image_name);
    if (img.empty()){
        std::cerr << "Image is not read" << std::endl;
        exit(0);
    }
    cv::Mat resized(INPUT_H, INPUT_W, CV_8UC3);
    cv::resize(img, resized, resized.size(), 0, 0, cv::INTER_AREA);

    float b, g, r;
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        b = resized.at<cv::Vec3b>(i)[0] / 255.0;
        b = (b - means[0])/stds[0];
        data[i] = b;

        g = resized.at<cv::Vec3b>(i)[1] / 255.0;
        g = (g - means[1])/stds[1];
        data[i + INPUT_H * INPUT_W] = g;

        r = resized.at<cv::Vec3b>(i)[2] / 255.0;
        r = (r - means[2])/stds[2];
        data[i + 2 * INPUT_H * INPUT_W] = r;
    }
}


float euclidean(float* feat1, float* feat2, int len){
    float sum = 0.0;
    for (int i = 0; i < len; i++){
        sum += pow(feat1[i] - feat2[i], 2.0);
    }
    return sqrt(sum);
}


int main(int argc, char** argv){
    int INPUT_C = 3;
    int INPUT_H = 224;
    int INPUT_W = 224;
    int OUTPUT_COUNT = 1;
    unsigned int batch_size = 1;
    std::string weight_path = "../resources/multigrain.wts";
    std::string img_name1("../image/dog1.jpg"); 
    std::string img_name2("../image/dog2.jpg"); 
    size_t inputSize = INPUT_C * INPUT_H * INPUT_W;
    float* img1 = new float[batch_size * inputSize];
    float* img2 = new float[batch_size * inputSize];
    
    process_image(img1, img_name1, INPUT_H, INPUT_W);
    process_image(img2, img_name2, INPUT_H, INPUT_W);

    MultiGrain *network = new MultiGrain(weight_path, INPUT_C, INPUT_H, INPUT_W, OUTPUT_COUNT, batch_size);
    
    auto start = std::chrono::system_clock::now();
    network->Init();
    network->DoInference(img1);
    float* feat1 = network->GetOutputPointerCPU();

    network->Init();
    network->DoInference(img2);
    float* feat2 = network->GetOutputPointerCPU();
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    auto outputDims = network->GetTensorDims(OUTPUT_BLOB_NAME);
    size_t outSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) outSize *= outputDims.d[i];

    float score = 1 - euclidean(feat1, feat2, outSize);
    std::cout << "Score: " << score << std::endl; 

    delete network;
    return 0;
}
