// Fast evaluation of our main network.

#include <iostream>
#include <chrono>
#include "fast_eval.h"

CudaBuffer::CudaBuffer(size_t length)
	: length(length)
{
	CUDA_CHECK(cudaMallocManaged(&contents, sizeof(float) * length));
}

CudaBuffer::~CudaBuffer() {
	CUDA_CHECK(cudaFree(contents));
}

CudaTensor::CudaTensor() {
	CUDA_CHECK(cudnnCreateTensorDescriptor(&desc));
}

CudaTensor::~CudaTensor() {
	CUDA_CHECK(cudnnDestroyTensorDescriptor(desc));
}

void CudaTensor::set_dimensions(std::tuple<int, int, int, int> dimensions) {
	CUDA_CHECK(cudnnSetTensor4dDescriptor(
		desc,
		cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
		cudnnDataType_t::CUDNN_DATA_FLOAT,
		std::get<0>(dimensions),
		std::get<1>(dimensions),
		std::get<2>(dimensions),
		std::get<3>(dimensions)
	));
}

AtaxxNetworkEvaluator::AtaxxNetworkEvaluator(int batch_size, int width, int height, int channels)
	: batch_size(batch_size),
	  width(width),
	  height(height),
	  channels(channels),
	  input_buffer(batch_size * width * height * channels),
	  scratch_buffer(batch_size * width * height * channels),
	  weights_buffer(batch_size * width * height * channels),
	  workspace_buffer(1<<20)
{
	CUDA_CHECK(cudnnCreate(&cudnn_handle));

	input_tensor.set_dimensions(std::tuple{batch_size, channels, width, height});
	scratch_tensor.set_dimensions(std::tuple{batch_size, 1, width, height});

	cudnnConvolutionDescriptor_t convolution_desc;
	CUDA_CHECK(cudnnCreateConvolutionDescriptor(&convolution_desc));
	CUDA_CHECK(cudnnSetConvolution2dDescriptor(
		convolution_desc,
		1, 1,
		1, 1,
		1, 1,
		CUDNN_CONVOLUTION,
		cudnnDataType_t::CUDNN_DATA_FLOAT
	));


	cudnnFilterDescriptor_t filter_desc;
	CUDA_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
	CUDA_CHECK(cudnnSetFilter4dDescriptor(
		filter_desc,
		cudnnDataType_t::CUDNN_DATA_FLOAT,
		cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
		batch_size, channels, 3, 3
	));

//	cudnnConvolutionFwdAlgo_t convolution_algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
	cudnnConvolutionFwdAlgo_t convolution_algo;
	CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(
		cudnn_handle, input_tensor.desc, filter_desc, convolution_desc, scratch_tensor.desc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algo
	));
	std::cout << "Convolution algorithm: " << convolution_algo << std::endl;

	size_t required_bytes;
	CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
		cudnn_handle,
		input_tensor.desc,
		filter_desc,
		convolution_desc,
		scratch_tensor.desc,
		convolution_algo,
		&required_bytes
	));
	std::cout << "Workspace size: " << required_bytes << std::endl;

/*
	int out_n, out_c, out_h, out_w;
	CUDA_CHECK(cudnnGetConvolution2dForwardOutputDim(
		convolution_desc, input_tensor.desc, filter_desc,
		&out_n, &out_c, &out_h, &out_w
	));
	std::cout << "Got: " << out_n << " " << out_c << " " << out_h << " " << out_w << std::endl;
*/

	float alpha = 1, beta = 0;

	for (int i = 0; i < 10; i++) {
		auto start = std::chrono::high_resolution_clock::now();
		CUDA_CHECK(cudnnConvolutionForward(
			cudnn_handle,
			&alpha,
			input_tensor.desc, input_buffer.contents,
			filter_desc, weights_buffer.contents,
			convolution_desc,
			convolution_algo,
			workspace_buffer.contents,
			workspace_buffer.length * sizeof(float),
			&beta,
			scratch_tensor.desc, scratch_buffer.contents
		));
		cudaDeviceSynchronize();
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = stop - start;
		std::cout << "Elapsed ms: " << elapsed.count() * 1e3 << std::endl;
	}

	cudnnDestroyConvolutionDescriptor(convolution_desc);
	cudnnDestroyFilterDescriptor(filter_desc);
}

AtaxxNetworkEvaluator::~AtaxxNetworkEvaluator() {
}

