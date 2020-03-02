#include <state_estimation/measurement_models/linear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

LinearMeasurementModel::LinearMeasurementModel(uint16_t n, uint16_t k)
    : MeasurementModel::MeasurementModel(n, k)
    , C_(Eigen::MatrixXd::Zero(meas_dims_, state_dims_)) {}

Eigen::MatrixXd LinearMeasurementModel::C() const {
    return C_;
}

void LinearMeasurementModel::setC(const Eigen::MatrixXd& new_C) {
    assert(new_C.rows() == meas_dims_ && new_C.cols() == state_dims_);
    C_ = new_C;
}

}  // namespace measurement_models
}  // namespace state_estimation
