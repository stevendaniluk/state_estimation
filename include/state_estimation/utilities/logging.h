#pragma once

#include <Eigen/Core>
#include <iostream>
#include <string>

namespace state_estimation {

// printMatrix
//
// Prints an Eigen matrix with a clean format. Will print in a single row when the matrix
// is a vector (i.e. columns=1).
//
// @param mat: Matrix to print
template <typename Real, int M, int N>
std::string printMatrix(const Eigen::Matrix<Real, M, N>& mat);

}  // namespace state_estimation

#include "logging.hpp"
