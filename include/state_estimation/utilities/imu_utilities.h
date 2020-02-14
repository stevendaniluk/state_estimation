#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// predictAccelMeasurement
//
// Computes the expected measurement from an accelerometer attached to a rigid body.
//
// @param a: Linear acceleration of body in the state frame
// @param w: Angular velocity of the body in the state frame
// @param tf: Transform from the state frame to the imu frame
// @param include_gravity: When true the predicted acceleration will have gravity added
// @param orientation: Orientation of the body in the state frame
// @param g: Gravitational acceleration to use
// @return: Predicted measurement [AX, AY, AZ]
Eigen::Matrix<double, 3, 1> predictAccelMeasurement(
    const Eigen::Vector3d& a, const Eigen::Vector3d& w, const Eigen::Isometry3d& tf,
    bool include_gravity = false, Eigen::Quaterniond orientation = Eigen::Quaterniond(),
    double g = 9.80665);

// predictGyroMeasurement
//
// Computes the expected measurement from a gyroscope attached to a rigid body.
//
// @param w: Angular velocity of the body in the state frame
// @param R: Rotation from the state frame to the imu frame
// @return: Predicted measurement [VPHI, VTHETA, VPSI]
Eigen::Matrix<double, 3, 1> predictGyroMeasurement(const Eigen::Vector3d& w,
                                                   const Eigen::Matrix3d& R);

// accelMeasurementJacobian
//
// Generates the Jacobian matrix for an acceleromete measurement accounting for a constant
// transformation between the measurement and state frames.
//
// The state vector is assumed to have the order:
//   [ax, ay, az, phi, theta, psi, phi_dot, theta_dot, psi_dot]
// While the measurement is assumed to have the order:
//   [ax, ay, ax]
//
// @param w: Current state angular rates
// @param tf: Transform from the state frame to the measurement frame
// @param include_gravity: When true the change in acceleration due to orientation will be computed
//                         from the effect of gravity
// @param rpy: Current state fixed axis roll, pitch, and yaw angles
// @param g: Gravitational acceleration to use
// @return: Measurement Jacobian matrix
Eigen::Matrix<double, 3, 9> accelMeasurementJacobian(const Eigen::Vector3d& w,
                                                     const Eigen::Isometry3d& tf,
                                                     bool include_gravity = true,
                                                     Eigen::Vector3d rpy = Eigen::Vector3d::Zero(),
                                                     double g = 9.80665);

// gyroMeasurementJacobian
//
// Generate the Jacobian matrix for a gyroscope measurement accounting for a constant
// transformation between the measurement and state frames.
//
// The state vector is assumed to have the order:
//   [phi_dot, theta_dot, psi_dot]
// While the measurement is assumed to have the order:
//   [phi_dot, theta_dot, psi_dot]
//
// @param R: Rotation from the state frame to the measurement frame
// @return: Measurement Jacobian matrix
Eigen::Matrix<double, 3, 3> gyroMeasurementJacobian(const Eigen::Matrix3d& R);

}  // namespace state_estimation
