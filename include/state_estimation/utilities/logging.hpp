#include <iostream>
#include <sstream>

namespace state_estimation {

template <typename Real, int M, int N>
std::string printMatrix(const Eigen::Matrix<Real, M, N>& mat) {
    std::ostringstream stream;
    stream << ((mat.cols() > 1) ? mat : mat.transpose())
                  .format(Eigen::IOFormat(6, 0, ", ", "\n", "[", "]"));
    return stream.str();
}

}  // namespace state_estimation
