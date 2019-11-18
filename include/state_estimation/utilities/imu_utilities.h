#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// predictImuMeasurement
//
// Computes the expected measurement from an imu attached to a rigid body.
//
// @param a: Linear acceleration of body in the state frame
// @param w: Angular velocity of the body in the state frame
// @param tf: Transform from the state frame to the imu frame
// @param include_gravity: When true the predicted acceleration will have gravity added
// @param orientation: Orientation of the body in the state frame
// @param g: Gravitational acceleration to use
// @return: Predicted measurement [AX, AY, AZ, VPHI, VTHETA, VPSI]
Eigen::Matrix<double, 6, 1> predictImuMeasurement(
    const Eigen::Vector3d& a, const Eigen::Vector3d& w, const Eigen::Isometry3d tf,
    bool include_gravity = false, Eigen::Quaterniond orientation = Eigen::Quaterniond(),
    double g = 9.80665);

// imuMeasurementJacobian
//
// Generates the Jacobian matrix for an imu measurement accounting for offsets between the
// measurement and state frames.
//
// The imu measurement is [ax, ay, az, phi_dot, theta_dot, psi_dot].
//
// This assumes the transform between the sensor and the state frane is constant.
//
// The state vector is assumed to have the order:
//   [ax, ay, az, phi, theta, psi, phi_dot, theta_dot, psi_dot]
// While the measurement is assumed to have the order:
//   [ax, ay, ax, phi_dot, theta_dot, psi_dot]
//
// Thus the Jacobian elements will be:
// [  dax/dax,     ...     dax/dphi,    ...    dax/dpsi_dot   ]
// [  day/dax,     ...     day/dphi,    ...    day/dpsi_dot   ]
// [     ...       ...       ...        ...        ...        ]
// [dpsi_dot/dax,  ...  dpsi_dot/dphi,  ...  dpsi_dot/dpsi_dot]
//
// @param w: Current state angular rates
// @param tf: Transform from the state frame to the measurement frame
// @param include_gravity: When true the change in acceleration due to orientation will be computed
//                         from the effect of gravity
// @param rpy: Current state fixed axis roll, pitch, and yaw angles
// @param g: Gravitational acceleration to use
// @return: Measurement Jacobian matrix
Eigen::Matrix<double, 6, 9> imuMeasurementJacobian(const Eigen::Vector3d& w,
                                                   const Eigen::Isometry3d& tf,
                                                   bool include_gravity = true,
                                                   Eigen::Vector3d rpy = Eigen::Vector3d::Zero(),
                                                   double g = 9.80665);

}  // namespace state_estimation
