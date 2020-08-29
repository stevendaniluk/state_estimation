#pragma once

#include <Eigen/Core>

namespace state_estimation {

// skew
//
// Skew operator that produces the cross product matrix for a vector:
//   [ 0  -Vz,  Vy]
//   [ Vz   0, -Vx]
//   [-Vy, Vx,   0]
//
// @param v: Vector
// @return: Cross product matrix
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

}  // namespace state_estimation
