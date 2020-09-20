#pragma once

#include <Eigen/Core>
#include <vector>

namespace state_estimation {

// Defines the vector indices used for 6DOF velocity estimation that utilizes an IMU.

namespace six_d_rates {

namespace state {
// Linear velocity
static const int VX = 0;
static const int VY = 1;
static const int VZ = 2;

// Linear Acceleration
static const int AX = 3;
static const int AY = 4;
static const int AZ = 5;

// Angular velocity
static const int VPHI = 6;
static const int VTHETA = 7;
static const int VPSI = 8;

// Gravity vector
static const int GX = 9;
static const int GY = 10;
static const int GZ = 11;

// Accelerometer biases
static const int B_AX = 12;
static const int B_AY = 13;
static const int B_AZ = 14;

// Gyro biases
static const int B_WX = 15;
static const int B_WY = 16;
static const int B_WZ = 17;

static const int DIMS = 18;
}  // namespace state

// The methods below define the addition, subtraction, and scaling operations for the state vector
// that will handle orientation range/normalization

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

}  // namespace six_d_rates

}  // namespace state_estimation
