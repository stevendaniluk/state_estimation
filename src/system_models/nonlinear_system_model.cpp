#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {
namespace system_models {

NonlinearSystemModel::NonlinearSystemModel(uint32_t n, uint32_t m, bool compute_jacobian,
                                           bool update_covariance)
    : SystemModel::SystemModel(n, m)
    , compute_jacobian_(compute_jacobian)
    , update_covariance_(update_covariance)
    , x_pred_(Eigen::VectorXd::Zero(n))
    , G_(Eigen::MatrixXd::Zero(n, n)) {}

Eigen::VectorXd NonlinearSystemModel::g() const {
    return x_pred_;
}

Eigen::MatrixXd NonlinearSystemModel::G() const {
    return G_;
}

}  // namespace system_models
}  // namespace state_estimation
