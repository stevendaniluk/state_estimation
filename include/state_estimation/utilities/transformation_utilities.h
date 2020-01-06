#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// transformLinearAcceleration
//
// Transforms an acceleration vector between two frames.
//
// This does not consider any coriolis forces.
//
// @param a: Acceleration vector
// @param w: Angular velocity vector
// @param tf: Transformation from the source frame to the target frame
// @return: Acceleration transformed to the target frame
Eigen::Vector3d transformLinearAcceleration(const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                                            const Eigen::Isometry3d& tf);

// transformLinearVelocity
//
// Transforms a linear velocity vector between two frames.
//
// @param v: Velocity vector to transform
// @param w: Angular velocity of body
// @param tf: Transform from the source frame to the target frame
// @return: Linear velocity transformed to target frame
Eigen::Vector3d transformLinearVelocity(const Eigen::Vector3d& v, const Eigen::Vector3d& w,
                                        const Eigen::Isometry3d& tf);

// transformAngularVelocity
//
// Transforms an angular velocity vector between two frames.
//
// @param w: Angular velocity vector
// @param R: Rotation from the source frame to the target frame
// @return: Angular velocity transformed to the target frame
Eigen::Vector3d transformAngularVelocity(const Eigen::Vector3d& w, const Eigen::Matrix3d& R);

// transformCovariance
//
// Rotates a 3D covariance matrix.
//
// @param cov: Covariance matrix to transform
// @param R: Rotation from the source frame to the target frame
// @return Covariance matrix transformed to the target frame
Eigen::Matrix3d transformCovariance(const Eigen::Matrix3d& cov, const Eigen::Matrix3d& R);

}  // namespace state_estimation
