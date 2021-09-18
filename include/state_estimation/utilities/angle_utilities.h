#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

namespace state_estimation {

// constrainAngle
//
//
// @param angle: Angle to constraint [rad]
// @return: Angle in the range [-PI, PI]
inline double constrainAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2 * M_PI;
    }

    return angle;
}

inline Eigen::VectorXd constrainAngle(const Eigen::VectorXd& angle) {
    Eigen::VectorXd result(angle.size());
    for (size_t i = 0; i < angle.size(); ++i) {
        result(i) = constrainAngle(angle(i));
    }

    return result;
}

// angleDifference
//
// Computes the shortest difference between two angles defined on the interval [-pi, pi].
//
// Example: theta1=0.9pi, theta2=-0.9pi, result=-0.2pi
//
// @param theta1: First angle [rad]
// @param theta2: Second angle [rad]
// @return: theta1 - theta 2, confined to the range [-pi, pi] [rad]
inline double angleDifference(double theta1, double theta2) {
    double dtheta = theta1 - theta2;
    dtheta = fmod(dtheta, 2 * M_PI);
    return fmod(2 * dtheta, 2 * M_PI) - dtheta;
}

inline Eigen::VectorXd angleDifference(const Eigen::VectorXd& theta1,
                                       const Eigen::VectorXd& theta2) {
    assert(theta1.size() == theta2.size());

    Eigen::VectorXd diff(theta1.size());
    for (size_t i = 0; i < theta1.size(); ++i) {
        diff(i) = angleDifference(theta1(i), theta2(i));
    }

    return diff;
}

// weightedQuaternion
//
// Computes a weighted average of quaternions.
//
// This is based on the formulation in:
//   Quaternion Averaging - F. Landis Markley, et al
//
// @param w: Weight for each quaternion
// @param q_set: Quaternions to average
// @return: Weighted average quaternion
inline Eigen::Quaterniond weightedQuaternion(
    const Eigen::VectorXd& w,
    const std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>& q_set) {
    assert(w.size() == q_set.size());

    // Form the weighting matrix
    Eigen::Matrix<double, 4, 4> m = Eigen::Matrix<double, 4, 4>::Zero();
    for (size_t i = 0; i < w.size(); ++i) {
        m += w(i) * (q_set[i].coeffs() * q_set[i].coeffs().transpose());
    }

    // Average quaternion is formed from the largest eigen vector
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4>> solver(m,
                                                                      Eigen::ComputeEigenvectors);
    Eigen::Vector4d vec = solver.eigenvectors().col(3);
    return Eigen::Quaterniond(vec(3), vec(0), vec(1), vec(2));
}

// weightedAngleSum
//
// Computes the weighted sum of a set of angles.
//
// This uses the mean of circular quantities technique, which converts each angle to cartesian
// coordinates then computes the weighted sum in Cartesian coordinates. Consequently, this
// implicitely normalizes the weight values to sum to 1.
//
// @param w: Weight for each angle
// @param thetas: Angles to average
// @return: Weighted average of
inline double weightedAngleSum(const Eigen::VectorXd& w, const Eigen::VectorXd& thetas) {
    assert(w.size() == thetas.size());

    double sin_sum = (w.array() * thetas.array().sin()).sum();
    double cos_sum = (w.array() * thetas.array().cos()).sum();
    return std::atan2(sin_sum, cos_sum);
}

}  // namespace state_estimation
