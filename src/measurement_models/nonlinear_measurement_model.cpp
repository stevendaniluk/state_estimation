#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

NonlinearMeasurementModel::NonlinearMeasurementModel(uint32_t n, uint32_t k, bool compute_jacobian)
    : MeasurementModel::MeasurementModel(n, k)
    , compute_jacobian_(compute_jacobian)
    , z_pred_(Eigen::VectorXd::Zero(k))
    , H_(Eigen::MatrixXd::Zero(k, n)) {}

Eigen::VectorXd NonlinearMeasurementModel::h() const {
    return z_pred_;
};

Eigen::MatrixXd NonlinearMeasurementModel::H() const {
    return H_;
}

}  // namespace measurement_models
}  // namespace state_estimation
