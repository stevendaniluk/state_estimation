#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {
namespace system_models {

NonlinearSystemModel::NonlinearSystemModel(uint16_t n, uint16_t m, uint16_t p,
                                           bool compute_jacobian, bool update_covariance)
    : SystemModel::SystemModel(n, m, p)
    , compute_jacobian_(compute_jacobian)
    , update_covariance_(update_covariance)
    , x_pred_(Eigen::VectorXd::Zero(state_dims_))
    , G_(Eigen::MatrixXd::Zero(state_dims_, state_dims_)) {}

Eigen::VectorXd NonlinearSystemModel::g() const {
    return x_pred_;
}

Eigen::MatrixXd NonlinearSystemModel::G() const {
    return G_;
}

}  // namespace system_models
}  // namespace state_estimation
