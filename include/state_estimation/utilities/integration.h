#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// deltaQuaternion
//
// Integrates angular rates to form a delta rotation over a time interval.
//
// @param dt: Time interval to project forward
// @param w: Angular velocity
Eigen::Quaterniond deltaQuaternion(double dt, const Eigen::Vector3d& w);

// integrate
//
// @param dt: Time interval to integrate over
// @param q: Initial orientation
// @param w: Angular velocity
// @return: Final orientation
Eigen::Quaterniond integrate(double dt, const Eigen::Quaterniond& q, const Eigen::Vector3d& w);

// integrateVelocityRK4
//
// Integrates velocity from acceleration using the RK4 method.
//
// @param dt: Time interval to integrate over
// @param a: Linear acceleration
// @param w: Angular velocity
// @param v: Velocity to increment (i.e. dv is computed, v + dv is returned)
void integrateVelocityRK4(double dt, const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                          Eigen::Vector3d* v);

// integratePositionRK4
//
// Integrates position from velocity acceleration using the RK4 method.
//
// @param dt: Time interval to integrate over
// @param v_i: Initial velocity
// @param a: Linear acceleration
// @param w: Angular velocity
// @param v: Velocity to increment (i.e. dv is computed, v + dv is returned)
void integratePositionRK4(double dt, Eigen::Vector3d v_i, const Eigen::Vector3d& a,
                          const Eigen::Vector3d& w, Eigen::Vector3d* p, Eigen::Vector3d* v);

}  // namespace state_estimation
