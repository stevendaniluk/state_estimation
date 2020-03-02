#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

NonlinearMeasurementModel::NonlinearMeasurementModel(uint16_t n, uint16_t k, bool compute_jacobian)
    : MeasurementModel::MeasurementModel(n, k)
    , compute_jacobian_(compute_jacobian)
    , z_pred_(Eigen::VectorXd::Zero(meas_dims_))
    , H_(Eigen::MatrixXd::Zero(meas_dims_, state_dims_)) {}

Eigen::VectorXd NonlinearMeasurementModel::h() const {
    return z_pred_;
};

Eigen::MatrixXd NonlinearMeasurementModel::H() const {
    return H_;
}

}  // namespace measurement_models
}  // namespace state_estimation
