#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/system_models/six_d_rates.h>
#include <state_estimation/utilities/integration.h>
#include <state_estimation/utilities/math_utilities.h>

namespace state_estimation {
namespace system_models {

using namespace six_d_rates;

SixDRates::SixDRates(bool compute_jacobian, bool update_covariance)
    : NonlinearSystemModel::NonlinearSystemModel(state::DIMS, 0, 15, compute_jacobian,
                                                 update_covariance) {
    // Diagonal entries of Jacobian will be 1
    G_ = Eigen::MatrixXd::Identity(G_.rows(), G_.cols());
}

void SixDRates::myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) {
    x_pred_ = x;

    // Need some helper constants
    const double dt_sq = pow(dt, 2);
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d g_i = x.segment(state::GX, 3);
    const Eigen::Vector3d w_i = x.segment(state::VPHI, 3);
    const Eigen::Quaterniond q_delta = deltaQuaternion(dt, w_i);

    // Integrate acceleration
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    integrateVelocityRK4(dt, x.segment(state::AX, 3), x.segment(state::VPHI, 3), &dv);

    x_pred_(state::VX) += stateUsage()[state::VX] * dv.x();
    x_pred_(state::VY) += stateUsage()[state::VY] * dv.y();
    x_pred_(state::VZ) += stateUsage()[state::VZ] * dv.z();

    // Rotate gravity
    if (stateUsage()[state::GX] && stateUsage()[state::GY] && stateUsage()[state::GZ]) {
        x_pred_.segment(state::GX, 3) = q_delta * g_i;
    }

    if (compute_jacobian_) {
        // Update the entries of our Jacobian, we don't need to handle logic for active
        // states/controls since the filters take care of grabbing the relevant sub matrix

        // Linear velocity wrt linear acceleration
        G_(state::VX, state::AX) = dt;
        G_(state::VY, state::AY) = dt;
        G_(state::VZ, state::AZ) = dt;

        // Gravity with respect to angular velocity
        G_.block<3, 3>(state::GX, state::VPHI) =
            -q_delta.w() * dt * skew(g_i) +
            0.5 * dt_sq *
                (w_i.transpose() * g_i * I3 + w_i * g_i.transpose() - g_i * w_i.transpose());

        // Gravity with respect to itself
        G_.block<3, 3>(state::GX, state::GX) =
            (pow(q_delta.w(), 2) + q_delta.vec().transpose() * q_delta.vec()) * I3 +
            q_delta.vec() * q_delta.vec().transpose() + 2 * q_delta.w() * skew(q_delta.vec());
    }

    if (update_covariance_) {
        // Update the process and control noise Jacobians, we don't need to handle logic for active
        // states/controls since the filters take care of grabbing the relevant sub matrix

        // Linear velocity wrt linear jerk
        P_(state::VX, 0) = 0.5 * dt_sq;
        P_(state::VY, 1) = 0.5 * dt_sq;
        P_(state::VZ, 2) = 0.5 * dt_sq;

        // Linear velocity wrt linear bias jerk
        P_(state::VX, 3) = 0.5 * dt_sq;
        P_(state::VY, 4) = 0.5 * dt_sq;
        P_(state::VZ, 5) = 0.5 * dt_sq;

        // Linear acceleration wrt linear jerk
        P_(state::AX, 0) = dt;
        P_(state::AY, 1) = dt;
        P_(state::AZ, 2) = dt;

        // Linear acceleration wrt linear bias jerk
        P_(state::AX, 3) = dt;
        P_(state::AY, 4) = dt;
        P_(state::AZ, 5) = dt;

        // Angular velocity wrt angular acceleration
        P_(state::VPHI, 6) = dt;
        P_(state::VTHETA, 7) = dt;
        P_(state::VPSI, 8) = dt;

        // Angular velocity wrt angular bias acceleration
        P_(state::VPHI, 9) = dt;
        P_(state::VTHETA, 10) = dt;
        P_(state::VPSI, 11) = dt;

        // Gravity wrt angular acceleration
        P_.block<3, 3>(state::GX, 12) =
            0.5 * pow(dt, 2) *
            (-q_delta.w() * skew(g_i) + dt * (w_i * g_i.transpose() + w_i.transpose() * g_i * I3) -
             0.5 * dt * w_i * g_i.transpose());
    }
}

Eigen::VectorXd SixDRates::addVectors(const Eigen::VectorXd& lhs,
                                      const Eigen::VectorXd& rhs) const {
    return six_d_rates::addState(lhs, rhs, stateUsage());
}

Eigen::VectorXd SixDRates::subtractVectors(const Eigen::VectorXd& lhs,
                                           const Eigen::VectorXd& rhs) const {
    return six_d_rates::subtractState(lhs, rhs, stateUsage());
}

Eigen::VectorXd SixDRates::weightedSum(const Eigen::VectorXd& w, const Eigen::MatrixXd& X) const {
    return six_d_rates::weightedSumOfStates(w, X, stateUsage());
}

}  // namespace system_models
}  // namespace state_estimation
