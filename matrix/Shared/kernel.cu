
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#define BLOCK_SIZE 16

__global__ void matmulGlobal(float* C, const float* A, const float* B, const int R) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	for (int i = 0; i < R; i++) {
		sum += A[row * R + i] * B[i * R + col];
	}
	C[row * R + col] = sum;
}


__global__ void matmulShared(float* C, const float* A, const float* B, const int R) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int localRow = threadIdx.y;
	int localCol = threadIdx.x;

	float sum = 0;
	__shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];
	for (int b = 0; b < (R / BLOCK_SIZE); b++) {
		int colA = b * blockDim.x + threadIdx.x;
		int rowB = b * blockDim.y + threadIdx.y;

		A_block[localRow][localCol] = A[row * R + colA];
		B_block[localRow][localCol] = B[rowB * R + col];
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; i++) {
			sum += A_block[localRow][i] * B_block[i][localCol];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	C[row * R + col] = sum;
}

void matmulCPU(float* C, const float* A, const float* B, const int R) {
	for (int row = 0; row < R; row++) {
		for (int col = 0; col < R; col++) {
			float sum = 0;
			for (int i = 0; i < R; i++) {
				sum += A[row * R + i] * B[i * R + col];
			}
			C[row * R + col] = sum;
		}
	}
}

// Helper function for using CUDA to matmul matrix in parallel.
cudaError_t matmulCUDA(float *C, const float *A, const float *B, const int R, const int USE_BLAS) {
	float* dev_A = NULL;
	float* dev_B = NULL;
	float* dev_C = NULL;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)& dev_C, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_A, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_B, R * R * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A, A, R * R * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_B, B, R * R * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	if (USE_BLAS) {
		cublasHandle_t handle;
		cublasCreate(&handle);

		float alpha = 1.0f;
		float beta = 0.0f;
		// use (AB)^T = B^T A^T to call BLAS without extra transpose
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, R, R, R, &alpha, dev_B, R, dev_A, R, &beta, dev_C, R);

		cublasDestroy(handle);
	} else {
		// Launch a kernel on the GPU with one thread for each element.
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(R / threadsPerBlock.x, R / threadsPerBlock.y);
		matmulShared << <numBlocks, threadsPerBlock >> > (dev_C, dev_A, dev_B, R);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matmulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matmulKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(C, dev_C, R * R * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_C);
	cudaFree(dev_A);
	cudaFree(dev_B);

	return cudaStatus;
}

float sumSquareError(float* A, float* B, const int size) {
	float sum = 0;
	float x = 0;
	for (int i = 0; i < size; i++) {
		x = (A[i] - B[i]);
		sum += x * x;
	}
	return sum;
}

int main(int argc, char* argv[]) {
	if (argc <= 1) {
		std::cout << "Usage: ./matrix R [USE_BLAS](default: yes) [DEBUG](default: no)" << std::endl;
		return EXIT_FAILURE;
	}

	const int R = std::stoi(argv[1]);
	int USE_BLAS = 1;
	int DEBUG = 0;

	if (argc >= 3) USE_BLAS = std::stoi(argv[2]);
	if (argc >= 4) DEBUG = std::stoi(argv[3]);

	float* A = new float[R * R];
	float* B = new float[R * R];
	float* C_CUDA = new float[R * R];
	float* C_CPU = new float[R * R];

	for (int i = 0; i < R * R; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	// matmulCUDA
	auto start_time_CUDA = std::chrono::system_clock::now();
	cudaError_t cudaStatus = matmulCUDA(C_CUDA, A, B, R, USE_BLAS);
	auto end_time_CUDA = std::chrono::system_clock::now();
	auto time_CUDA = std::chrono::duration_cast<std::chrono::microseconds>(end_time_CUDA - start_time_CUDA).count() / 1e6;
	double gflops_CUDA = 2 * pow(R, 3)/ pow(1024, 3) / time_CUDA;
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matmulCUDA failed!");
		return EXIT_FAILURE;
	}

	std::cout << "R = " << R << ", BLOCK_SIZE = " << BLOCK_SIZE << ", USE_BLAS = " << USE_BLAS << ", time_CUDA: " << time_CUDA << ", gflops_CUDA: " << gflops_CUDA;
	if (DEBUG) {
		// matmulCPU
		auto start_time_CPU = std::chrono::system_clock::now();
		matmulCPU(C_CPU, A, B, R);
		auto end_time_CPU = std::chrono::system_clock::now();
		auto time_CPU = std::chrono::duration_cast<std::chrono::microseconds>(end_time_CPU - start_time_CPU).count() / 1e6;
		double gflops_CPU = 2 * pow(R, 3) / pow(1024, 3) / time_CPU;

		std::cout << ", time_CPU: " << time_CPU << ", gflops_CPU: " << gflops_CPU;
		if (memcmp(C_CUDA, C_CPU, sizeof(float) * R * R) == 0) {
			std::cout << ", Equal Bytewise" << std::endl;
		} else {
			float sse = sumSquareError(C_CUDA, C_CPU, R * R);
			std::cout << ", Sum Square Error: " << sse << std::endl;
		}

		std::ofstream outA("A.txt");
		std::ofstream outB("B.txt");
		std::ofstream outC_CUDA("C_CUDA.txt");
		std::ofstream outC_CPU("C_CPU.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col++) {
				outA << A[row * R + col] << " ";
				outB << B[row * R + col] << " ";
				outC_CUDA << C_CUDA[row * R + col] << " ";
				outC_CPU << C_CPU[row * R + col] << " ";
			}
			outA << std::endl;
			outB << std::endl;
			outC_CUDA << std::endl;
			outC_CPU << std::endl;
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return EXIT_FAILURE;
	}

	delete[] A;
	delete[] B;
	delete[] C_CUDA;
	delete[] C_CPU;

	return EXIT_SUCCESS;
}
