#ifndef TENSORRT_BASE_H
#define TENSORRT_BASE_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "common.h"
#include "logger.h"
#include <chrono>
#include <stdexcept>


static Logger gLogger;

//!
//! \class TensorRT
//!
//! \brief A base class for TensorRT Inference
//!
//! This base class consists of common methods and members which are meant to be
//! used in derived classes for deep learning models that are wanted to be used 
//! in inference stage.
//!  
//! \warning Do not try to create an instance from this class since it is an abstract class
//!
class TensorRT {
public:
	//!
    //! \brief Constructor for TensorRT class
    //!
    //! This constructor is called in the constructor of the derived class as
    //! DerivedClassConstructor(derived_class_parameters): TensorRT(inputNames, outputNames, inputDims, batchSize, outputToCPU) {}
    //!
    //! \param inputNames A vector holding names of the model's inputs.
    //! \param outputNames A vector holding names of the model's outputs.
    //! \param inputDims A 2d vector holding the dimensions of the inputs.
    //! \param batchSize The value of the batch size.
    //! \param outputToCPU The boolean whether or not to move output to CPU from GPU.
    //!
	explicit TensorRT(std::vector<const char*> inputNames, std::vector<const char*> outputNames, 
                    std::vector<std::vector<int>> inputDims, unsigned int batchSize, bool outputToCPU);

	//!
    //! \brief Destructor for TensorRT class
    //!
    //! This destructor closes everything down, releases the stream and buffers.
    //!					
	~TensorRT();

	//!
    //! \brief Does necessary steps about plan file
    //!
    //! Loads the weights and serializes model to plan file if a cache file does not exists.
    //! Then, it deserializes plan file for inference. 
    //!
    //! \param weightPath The name of the weights file with .wts extension.
    //!
    //! \warning This function should be called before Init and DoInference methods.
    //!
    //! \return void
    //!
	void PrepareInference(std::string &weightPath);

	//!
    //! \brief Does memory related operations before inference
    //!
    //! Allocates memory for input and output tensors, and creates a cuda stream.
    //!
    //! \warning This function should be called before DoInference method.
    //!
    //! \return void
    //!
	void Init();

	//!
    //! \brief Starts the inference stage
    //!
    //! Creates a buffer consisting of input and output GPU pointers and does cuda member copy operations
    //!
	//! \param batchSize The value of batch size to be used in inference.
	//! \param inputs An array of float pointers holding the model's input arrays.
	//!
    //! \return void
    //!
	void DoInference(int batchSize, float* inputs[]);

	//!
    //! \brief Returns the wanted input gpu pointer
    //!
	//! \param index The index of the input gpu pointer.
	//!
    //! \return A float pointer pointing to GPU for the spesified input. 
    //!
	float *GetInputPointersGPU(int index)  { return mInputPointersGPU[index]; }

	//!
    //! \brief Returns the wanted output gpu pointer
    //!
	//! \param index The index of the output gpu pointer.
	//!
    //! \return A float pointer pointing to GPU for the spesified output. 
    //!
	float *GetOutputPointersGPU(int index) { return mOutputPointersGPU[index]; }

	//!
    //! \brief Returns the wanted output cpu pointer
    //!
	//! \param index The index of the output cpu pointer.
	//!
    //! \return A float pointer pointing to CPU for the spesified output. 
    //!
	float *GetOutputPointersCPU(int index) { return mOutputPointersCPU[index]; }

	//!
    //! \brief Returns the dimensions of the wanted tensor
    //!
	//! \param name The name of the tensor whose dimensions are wanted to be obtained.
	//!
    //! \return A Dims instance holding the dimensions of the tensor. 
    //!
	const nvinfer1::Dims GetTensorDims(const char *name);

	//!
    //! \brief Returns the data type of the wanted tensor
    //!
	//! \param name The name of the tensor whose data type is to be obtained.
	//!
    //! \return A DataType instance stating the data type of the tensor. 
    //!
	const nvinfer1::DataType GetTensorDataType(const char *name);

    //!
    //! \brief Returns the size of the data type of the tensor
    //!
	//! \param name The name of the tensor whose data type size is to be obtained.
	//!
    //! \return size_t value stating the data type size of the tensor. 
    //!
	const size_t GetTensorDataTypeSize(const char *name);

protected:
	//!
    //! \brief Defines the model layer by layer and serialize the model to plan file
    //!
    //! This method is a pure virtual function and should be implemented in the derived classes.
    //! The model's layers should be analyzed and the pipeline should be implemented accordingly.
    //!
    //! \param cachePath The path of the serialized engine that will be saved.
    //! \param weightMap A map of string an Weights to store the pretrained weights of the model.
    //! \param maxBatchSize The maximum batch size used while serializing the model.
    //! \param dt The datatype used (kFLOAT etc.).
    //!
    //! \return void
    //!
	virtual void SerializeEngine(char* cachePath, std::map<std::string, nvinfer1::Weights>& weightMap, 
                                nvinfer1::IBuilder* &builder, nvinfer1::IBuilderConfig* &config, 
                                nvinfer1::IHostMemory* &modelStream, unsigned int maxBatchSize, nvinfer1::DataType dt) = 0;

    //!
    //! \brief Deserializes the plan file saved by SerializeEngine method for inference
    //!
    //! \param cacheFile The path of the serialized engine to be deserialized.
    //!
    //! \return void
    //!
	void DeserializeEngine(std::ifstream& cacheFile);

	nvinfer1::IRuntime *mpRuntime = nullptr;                //!< The pointer to IRuntime necessary for inference
	nvinfer1::ICudaEngine *mpEngine = nullptr;              //!< The pointer to ICudaEngine necessary for inference
	nvinfer1::IExecutionContext *mpContext = nullptr;       //!< The pointer to IExecutionContext necessary for inference
	cudaStream_t mStream;                                  //!< Created cuda stream that is used in inference.
	std::vector<size_t> mNetworkInputSizes;                //!< A vector holding the sizes of inputs of the model.
	std::vector<size_t> mNetworkOutputSizes;               //!< A vector holding the sizes of outputs of the model.
	std::vector<float*> mInputPointersGPU;                 //!< A vector holding the input pointers which points to GPU.
	std::vector<float*> mOutputPointersGPU;                //!< A vector holding the output pointers which points to GPU.
	std::vector<float*> mOutputPointersCPU;                //!< A vector holding the output pointers which points to CPU.
	std::vector<const char*> mInputNames;                  //!< A vector holding the names of input tensors.
	std::vector<const char*> mOutputNames;                 //!< A vector holding the names of output tensors.
    std::vector<std::vector<int>> mInputDims;              //!< A 2d vector holding the dimensions of the inputs.
	unsigned int mNetworkBatchSize = 0;                    //!< The batch size used in inference.
	bool mOutputToCPU = false;                             //!< Boolean value stating whether output will be moved to CPU from GPU.
};
#endif // TENSORRT_BASE_H

