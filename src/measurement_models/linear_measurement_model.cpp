#include <state_estimation/measurement_models/linear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

LinearMeasurementModel::LinearMeasurementModel(uint32_t n, uint32_t k, bool compute_covariance)
    : MeasurementModel::MeasurementModel(n, k, compute_covariance)
    , C_(Eigen::MatrixXd::Zero(k, n)) {}

Eigen::MatrixXd LinearMeasurementModel::C() const {
    return C_;
}

void LinearMeasurementModel::setC(const Eigen::MatrixXd& new_C) {
    assert(new_C.rows() == C_.rows() && new_C.cols() == C_.cols());
    C_ = new_C;
}

}  // namespace measurement_models
}  // namespace state_estimation
