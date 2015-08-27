#include <iostream>
#include <boost/format.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace boost;

__global__ void multiplyKernel(const float* A, const float* B, float* C, const unsigned int R);
cudaError_t multiplyWithCuda(const float* A, const float* B, float* C, const unsigned int R);

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << format("Usage: %1% <R>") % argv[0] << endl;
		return EXIT_FAILURE;
	}

	const int R = stoi(argv[1]);

	float* A = new float[R * R];
	float* B = new float[R * R];
	float* C = new float[R * R];

	for (int i = 0; i < R * R; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	// Add vectors in parallel.
	cudaError_t cudaStatus = multiplyWithCuda(A, B, C, R);
	if (cudaStatus != cudaSuccess) {
		cerr << "multiplyWithCuda failed!" << endl;
		return EXIT_FAILURE;
	}

	cout << format("Result Matrix Pointer: %1%") % C << endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaDeviceReset failed!" << endl;
		return EXIT_FAILURE;
	}

	delete[] A;
	delete[] B;
	delete[] C;

	return EXIT_SUCCESS;
}

__global__ void multiplyKernel(const float* A, const float* B, float* C, const unsigned int R) {
	int row = blockIdx.x;
	int col = threadIdx.x;
	float sum = 0;

	for (int i = 0; i < R; i++) {
		sum += A[row * R + i] * B[row * R + i];
	}

	C[row * R + col] = sum;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyWithCuda(const float* A, const float* B, float* C, const unsigned int R) {
	float* devA = NULL;
	float* devB = NULL;
	float* devC = NULL;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devA, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devB, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devC, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devA, A, R * R * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devB, B, R * R * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	multiplyKernel << <R, R >> >(devA, devB, devC, R);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(C, devC, R * R * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return cudaStatus;
}
