// Fast evaluation of our main network.

#ifndef ATAXXZERO_NETWORK_EVALUATOR_H
#define ATAXXZERO_NETWORK_EVALUATOR_H

#include <tuple>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

// Copied from: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_CHECK(ans) do { _gpuAssert(static_cast<cudaError_t>((ans)), __FILE__, __LINE__); } while(0)

static inline void _gpuAssert(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "cudaCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

struct CudaBuffer {
	size_t length;
	float* contents;

	CudaBuffer(size_t length);
	~CudaBuffer();
};

struct CudaTensor {
	cudnnTensorDescriptor_t desc;

	CudaTensor();
	~CudaTensor();
	void set_dimensions(std::tuple<int, int, int, int> dimensions);
};

struct AtaxxNetworkEvaluator {
	cudnnHandle_t cudnn_handle;
	CudaBuffer input_buffer, scratch_buffer, weights_buffer, workspace_buffer;
	CudaTensor input_tensor, scratch_tensor, workspace_tensor;
	int batch_size, width, height, channels;

	AtaxxNetworkEvaluator(int batch_size, int width, int height, int channels);
	~AtaxxNetworkEvaluator();
};

#endif

