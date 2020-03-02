#pragma once

#include <Eigen/Core>
#include <vector>

namespace state_estimation {

// Defines the vector indices used for planer 2d states, controls, and measurements

namespace planer_2d {

// State vector indices.
namespace state {
static const int X = 0;
static const int Y = 1;
static const int VX = 2;
static const int VY = 3;
static const int AX = 4;
static const int AY = 5;
static const int PSI = 6;
static const int VPSI = 7;
static const int APSI = 8;

static const int DIMS = 9;
}  // namespace state

// The methods below define the addition, subtraction, and scaling operations for the state vector
// that will handle keeping the heading angle within the range [-pi, pi]

// addState
//
// @param usage: A bit field of which indices to sum (0=not used)
// @return: lhs + rhs
Eigen::VectorXd addState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                         const std::vector<uint8_t>& usage);

// subtractState
//
// @param usage: A bit field of which indices to subtract (0=not used)
// @return: lhs - rhs
Eigen::VectorXd subtractState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                              const std::vector<uint8_t>& usage);

// weightedSumOfStates
//
// Provides the operation to compute the weighted sum of a set of sttae vectors, i.e.
//   X' = sum_i(w * X_i)
//
// @param w: Scalar weight values
// @param X: Matrix of data to process, each column is a vector to scale by the weight
// @param usage: A bit field of which indices to sum (0=not used)
// @return: Weighted sum
Eigen::VectorXd weightedSumOfStates(const Eigen::VectorXd& w, const Eigen::MatrixXd& X,
                                    const std::vector<uint8_t>& usage);

}  // namespace planer_2d

}  // namespace state_estimation
