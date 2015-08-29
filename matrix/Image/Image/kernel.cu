#include <iostream>
#include <fstream>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <cuda_runtime.h>
#include "cutil_math.h"

using namespace std;
using namespace boost;

__global__ void multiplyKernel();
void logTime(const string& message);

// globals
texture<float4, cudaTextureType2D, cudaReadModeElementType> texA;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texB;
surface<void, cudaSurfaceType2D> surfC;

int main(int argc, char* argv[]) {
	if (argc != 2 && argc != 3) {
		cout << format("Usage: %1% <R> <save?>") % argv[0] << endl;
		return EXIT_FAILURE;
	}

	logTime("Launched");
	const int R = stoi(argv[1]);
	const int W = R / 4;
	float* A = new float[R * R];
	float* B = new float[R * R];
	float* C = new float[R * R];
	logTime("Host memory allocated");

	for (int i = 0; i < R * R; i++) {
		A[i] = rand();
		B[i] = rand();
	}
	logTime("Matrix randomly filled");

	try {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cudaArray* devA = NULL;
		cudaArray* devB = NULL;
		cudaArray* devC = NULL;
		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

		cudaStatus = cudaMallocArray(&devA, &channelDesc, W, R);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMallocArray(&devB, &channelDesc, W, R);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMallocArray(&devC, &channelDesc, W, R, cudaArraySurfaceLoadStore);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMalloc failed!");

		cudaStatus = cudaMemcpyToArray(devA, 0, 0, A, R * R * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

		cudaStatus = cudaMemcpyToArray(devB, 0, 0, B, R * R * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

		cudaBindTextureToArray(texA, devA);
		cudaBindTextureToArray(texB, devB);
		cudaBindSurfaceToArray(surfC, devC);

		logTime("Ready to launch kernel");
		timer t;
		multiplyKernel<<<R, W>>>();

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) throw runtime_error((format("multiplyKernel launch failed: %1%") % (cudaGetErrorString(cudaStatus))).str());

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw runtime_error((format("cudaDeviceSynchronize failed: %1%") % (cudaGetErrorString(cudaStatus))).str());

		logTime("Kernel finish");
		double gflops = pow(R, 3) / 536870912 / t.elapsed();
		cout << format("%1% GFLOPS / %2%s Computing Time") % gflops % t.elapsed() << endl;

		cudaStatus = cudaMemcpyFromArray(C, devC, 0, 0, R * R * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaMemcpy failed!");

		cudaFree(devA);
		cudaFree(devB);
		cudaFree(devC);

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) throw runtime_error("cudaDeviceReset failed!");

	} catch (const runtime_error& e) {
		cerr << e.what() << endl;
		return EXIT_FAILURE;
	}

	logTime("Back to host");
	cout << format("Result Matrix Pointer: %1$p") % C << endl;

	if (argc == 3) {
		ofstream outA("A.csv");
		ofstream outB("B.csv");
		ofstream outC("C.csv");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col++) {
				outA << A[row * R + col] << " ";
				outB << B[row * R + col] << " ";
				outC << C[row * R + col] << " ";
			}
			outA << endl;
			outB << endl;
			outC << endl;
		}
		logTime("Save finish");
	}

	delete[] A;
	delete[] B;
	delete[] C;

	logTime("Finish");
	return EXIT_SUCCESS;
}

__global__ void multiplyKernel() {
	int X = threadIdx.x;
	int Y = blockIdx.x;
	float4 sum;

	for (int i = 0; i < blockDim.x; i++) {
		float4 valueA = tex2D(texA, i, Y);
		sum += make_float4(
			dot(valueA, tex2D(texB, i, 4 * X)),
			dot(valueA, tex2D(texB, i, 4 * X + 1)),
			dot(valueA, tex2D(texB, i, 4 * X + 2)),
			dot(valueA, tex2D(texB, i, 4 * X + 3))
		);
	}

	surf2Dwrite(sum, surfC, 16 * X, Y);
}

void logTime(const string& message) {
	static timer t;
	cout << boost::format("[%1$.3f] %2%") % t.elapsed() % message << endl;
}
