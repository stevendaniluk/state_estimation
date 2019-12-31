#pragma once

#include <Eigen/Core>

namespace state_estimation {

// Defines the vector indices used for planer 2d states, controls, and measurements

namespace planer_2d {

// State vector indices.
//
// It is common for X, Y and PSI to correspond to the fixed frame, and all rates and accelerations
// to correspond to the inertial frame.
namespace state {
    static const int X = 0;
    static const int Y = 1;
    static const int VX = 2;
    static const int VY = 3;
    static const int AX = 4;
    static const int AY = 5;
    static const int PSI = 6;
    static const int VPSI = 7;

    static const int DIMS = 8;
}// end state namespace

// The methods below define the addition, subtraction, and scaling operations for the state vector
// that will handle keeping the heading angle within the range [-pi, pi]

// addState
//
// @return: lhs + rhs
Eigen::VectorXd addState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs);

// subtractState
//
// @return: lhs - rhs
Eigen::VectorXd subtractState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs);

// weightedSumOfStates
//
// Provides the operation to compute the weighted sum of a set of sttae vectors, i.e.
//   X' = sum_i(w * X_i)
//
// @param w: Scalar weight values
// @param X: Matrix of data to process, each column is a vector to scale by the weight
// @return: Weighted sum
Eigen::VectorXd weightedSumOfStates(const Eigen::VectorXd& w, const Eigen::MatrixXd& X);

}// end planer_2d namespace

} // end state_estimation namespace
