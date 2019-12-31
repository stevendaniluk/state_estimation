#pragma once

#include <Eigen/Core>

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
