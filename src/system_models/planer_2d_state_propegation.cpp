#include <state_estimation/system_models/planer_2d_state_propegation.h>
#include <state_estimation/utilities/angle_utilities.h>

namespace state_estimation {
namespace system_models {

using namespace planer_2d;

Planer2DStatePropegation::Planer2DStatePropegation(bool compute_jacobian)
    : NonlinearSystemModel::NonlinearSystemModel(state::DIMS, 0, compute_jacobian) {
    G_ = Eigen::MatrixXd::Identity(G_.rows(), G_.cols());
}

void Planer2DStatePropegation::myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                        double dt) {
    (void)u;
    updateNoControl(x, dt);
}

void Planer2DStatePropegation::myUpdateNoControl(const Eigen::VectorXd& x, double dt) {
    // Compute some constants that we will need
    const double cpsi = std::cos(x(state::PSI));
    const double spsi = std::sin(x(state::PSI));
    const double dt_sq = dt * dt;
    const double dt_cpsi = dt * cpsi;
    const double dt_spsi = dt * spsi;
    const double dt_sq_cpsi = dt_sq * cpsi;
    const double dt_sq_spsi = dt_sq * spsi;

    const double dt_vx_cpsi = dt_cpsi * x(state::VX);
    const double dt_vx_spsi = dt_spsi * x(state::VX);
    const double dt_vy_cpsi = dt_cpsi * x(state::VY);
    const double dt_vy_spsi = dt_spsi * x(state::VY);

    const double dt_sq_ax_cpsi = dt_sq_cpsi * x(state::AX);
    const double dt_sq_ax_spsi = dt_sq_spsi * x(state::AX);
    const double dt_sq_ay_cpsi = dt_sq_cpsi * x(state::AY);
    const double dt_sq_ay_spsi = dt_sq_spsi * x(state::AY);

    // Update our state prediction with a constant acceleration model
    x_pred_(state::X) =
        x(state::X) + dt_vx_cpsi + 0.5 * dt_sq_ax_cpsi - dt_vy_spsi - 0.5 * dt_sq_ay_spsi;
    x_pred_(state::Y) =
        x(state::Y) + dt_vx_spsi + 0.5 * dt_sq_ax_spsi + dt_vy_cpsi + 0.5 * dt_sq_ay_cpsi;
    x_pred_(state::VX) = x(state::VX) + dt * x(state::AX);
    x_pred_(state::VY) = x(state::VY) + dt * x(state::AY);
    x_pred_(state::AX) = x(state::AX);
    x_pred_(state::AY) = x(state::AY);
    x_pred_(state::PSI) = x(state::PSI) + dt * x(state::VPSI);
    x_pred_(state::VPSI) = x(state::VPSI);

    // Constrain out orientation to be in the interval [-pi, pi]
    x_pred_(state::PSI) = constrainAngle(x_pred_(state::PSI));

    // Update the entries of our Jacobian, these are all analytically derived
    if (compute_jacobian_) {
        G_(state::X, state::VX) = dt_cpsi;
        G_(state::X, state::VY) = -dt_spsi;
        G_(state::X, state::AX) = 0.5 * dt_sq_cpsi;
        G_(state::X, state::AY) = -0.5 * dt_sq_spsi;
        G_(state::X, state::PSI) =
            -dt_vx_spsi - 0.5 * dt_sq_ax_spsi - dt_vy_cpsi - 0.5 * dt_sq_ay_cpsi;

        G_(state::Y, state::VX) = dt_spsi;
        G_(state::Y, state::VY) = dt_cpsi;
        G_(state::Y, state::AX) = 0.5 * dt_sq_spsi;
        G_(state::Y, state::AY) = 0.5 * dt_sq_cpsi;
        G_(state::Y, state::PSI) =
            dt_vx_cpsi + 0.5 * dt_sq_ax_cpsi - dt_vy_spsi - 0.5 * dt_sq_ay_spsi;

        G_(state::VX, state::AX) = dt;
        G_(state::VY, state::AY) = dt;

        G_(state::PSI, state::VPSI) = dt;
    }
}

Eigen::VectorXd Planer2DStatePropegation::addVectors(const Eigen::VectorXd& lhs,
                                                     const Eigen::VectorXd& rhs) const {
    return planer_2d::addState(lhs, rhs);
}

Eigen::VectorXd Planer2DStatePropegation::subtractVectors(const Eigen::VectorXd& lhs,
                                                          const Eigen::VectorXd& rhs) const {
    return planer_2d::subtractState(lhs, rhs);
}

Eigen::VectorXd Planer2DStatePropegation::weightedSum(const Eigen::VectorXd& w,
                                                      const Eigen::MatrixXd& X) const {
    return planer_2d::weightedSumOfStates(w, X);
}

}  // namespace system_models
}  // namespace state_estimation
