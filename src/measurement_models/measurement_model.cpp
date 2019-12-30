#include <state_estimation/measurement_models/measurement_model.h>

namespace state_estimation {

MeasurementModel::MeasurementModel(uint32_t n, uint32_t k)
    : FilterModel::FilterModel(n)
    , meas_dims_(k) {
    cov_ = Eigen::MatrixXd::Zero(k, k);
}

void MeasurementModel::update(const Eigen::VectorXd& x) {
    assert(x.size() == state_dims_);
    myUpdate(x);
};

uint32_t MeasurementModel::measurementSize() const {
    return meas_dims_;
}

}  // namespace state_estimation
