#include <iostream>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace boost;

__global__ void multiplyKernel(const float* A, const float* B, float* C, const unsigned int R);
cudaError_t multiplyWithCuda(const float* A, const float* B, float* C, const unsigned int R);
void logTime(const string& message);

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << format("Usage: %1% <R>") % argv[0] << endl;
		return EXIT_FAILURE;
	}

	logTime("Launched");
	const int R = stoi(argv[1]);

	float* A = new float[R * R];
	float* B = new float[R * R];
	float* C = new float[R * R];

	logTime("Host memory allocated");

	for (int i = 0; i < R * R; i++) {
		A[i] = float(rand() % 100);
		B[i] = float(rand() % 100);
	}

	logTime("Matrix randomly filled");

	// Add vectors in parallel.
	cudaError_t cudaStatus = multiplyWithCuda(A, B, C, R);
	if (cudaStatus != cudaSuccess) {
		cerr << "multiplyWithCuda failed!" << endl;
		return EXIT_FAILURE;
	}

	logTime("Back to host");
	cout << format("Result Matrix Pointer: %1$x") % C << endl;

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

	logTime("Finish");

	return EXIT_SUCCESS;
}

__global__ void multiplyKernel(const float* A, const float* B, float* C, const unsigned int R) {
	int row = blockIdx.x;
	int col = threadIdx.x;
	float sum = 0;

	for (int i = 0; i < R; i++) {
		sum += A[row * R + i] * B[i * R + col];
	}

	C[row * R + col] = sum;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyWithCuda(const float* A, const float* B, float* C, const unsigned int R) {
	float* devA = NULL;
	float* devB = NULL;
	float* devC = NULL;
	cudaError_t cudaStatus;

	try {
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

		cudaStatus = cudaMalloc((void**)&devA, R * R * sizeof(float));
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&devB, R * R * sizeof(float));
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&devC, R * R * sizeof(float));
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMemcpy(devA, A, R * R * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

		cudaStatus = cudaMemcpy(devB, B, R * R * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

		logTime("Ready to launch kernel");
		timer t;
		multiplyKernel << <R, R >> >(devA, devB, devC, R);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) throw runtime_error((format("addKernel launch failed: %1%") % (cudaGetErrorString(cudaStatus))).str());

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw runtime_error((format("cudaDeviceSynchronize returned error code %1% after launching addKernel!") % cudaStatus).str());

		logTime("Kernel finish");
		double gflops = pow(R, 3) / 536870912 / t.elapsed();
		cout << format("%1% GFLOPS / %2%s Computing Time") % gflops % t.elapsed() << endl;

		cudaStatus = cudaMemcpy(C, devC, R * R * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

	} catch (const runtime_error& e) {
		cerr << e.what() << endl;
	}

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return cudaStatus;
}

void logTime(const string& message) {
	static timer t;
	cout << boost::format("[%1$.3f] %2%") % t.elapsed() % message << endl;
}