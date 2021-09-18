#include <state_estimation/utilities/integration.h>
#include <iostream>

namespace state_estimation {

Eigen::Quaterniond deltaQuaternion(double dt, const Eigen::Vector3d& w) {
    const double w_norm = w.norm();
    if (w_norm > 1e-9) {
        const double theta_half = 0.5 * dt * w_norm;
        const double c = std::sin(theta_half) / w_norm;

        return Eigen::Quaterniond(std::cos(theta_half), c * w.x(), c * w.y(), c * w.z());
    } else {
        return Eigen::Quaterniond::Identity();
    }
}

Eigen::Quaterniond integrate(double dt, const Eigen::Quaterniond& q, const Eigen::Vector3d& w) {
    return q * deltaQuaternion(dt, w);
}

void integrateVelocityRK4(double dt, const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                          Eigen::Vector3d* v) {
    // ODE is:
    //   [v'] = [R(t) * a]

    // Delta rotations at the mid and end points of the step size for rotating the velocity and
    // acceleration vectors
    const Eigen::Quaterniond q_mid = deltaQuaternion(0.5 * dt, w);
    const Eigen::Quaterniond q_end = deltaQuaternion(dt, w);

    // RK4 for the velocity
    const Eigen::Vector3d k1_v = a;
    const Eigen::Vector3d k2_v = q_mid * a;
    const Eigen::Vector3d k3_v = k2_v;
    const Eigen::Vector3d k4_v = q_end * a;
    (*v) += dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;
}

void integratePositionRK4(double dt, Eigen::Vector3d v_i, const Eigen::Vector3d& a,
                          const Eigen::Vector3d& w, Eigen::Vector3d* p, Eigen::Vector3d* v) {
    // System of ODEs is:
    //   [p'] = [R(t) * v]
    //   [v'] = [R(t) * a]

    // Delta rotations at the mid and end points of the step size for rotating the velocity and
    // acceleration vectors
    const Eigen::Quaterniond q_mid = deltaQuaternion(0.5 * dt, w);
    const Eigen::Quaterniond q_end = deltaQuaternion(dt, w);

    // RK4 for the velocity
    const Eigen::Vector3d k1_v = a;
    const Eigen::Vector3d k2_v = q_mid * a;
    const Eigen::Vector3d k3_v = k2_v;
    const Eigen::Vector3d k4_v = q_end * a;
    (*v) += dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;

    // RK4 for the position
    // k2 is equal to k3 for this case, so we won't compute k3
    const Eigen::Vector3d k1_p = v_i;
    const Eigen::Vector3d k2_p = q_mid * (v_i + 0.5 * dt * k1_v);
    const Eigen::Vector3d k3_p = q_mid * (v_i + 0.5 * dt * k2_v);
    const Eigen::Vector3d k4_p = q_end * (v_i + dt * k3_v);
    Eigen::Vector3d p0 = *p;
    (*p) += dt * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6;
}

}  // namespace state_estimation
