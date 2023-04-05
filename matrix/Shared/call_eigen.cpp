#include <Eigen/Dense>

using namespace Eigen;

void matmulEigen(float* C, const float* A, const float* B, const int R) {
	Eigen::setNbThreads(Eigen::nbThreads() / 2);
	Eigen::Map<const Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>> eigen_A(A, R, R);
	Eigen::Map<const Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>> eigen_B(B, R, R);
	Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>> eigen_C(C, R, R);
	eigen_C.noalias() = eigen_A * eigen_B;
}
