#include <state_estimation/measurement_models/measurement_model.h>

namespace state_estimation {

MeasurementModel::MeasurementModel(uint32_t n, uint32_t k)
    : Q_(Eigen::MatrixXd::Zero(k, k))
    , state_dims_(n)
    , meas_dims_(k)
    , tf_(Eigen::Isometry3d::Identity()) {}

void MeasurementModel::update(const Eigen::VectorXd& x) {
    assert(x.size() == state_dims_);
    myUpdate(x);
};

Eigen::MatrixXd MeasurementModel::Q() const {
    return Q_;
}

void MeasurementModel::setQ(const Eigen::MatrixXd& new_Q) {
    assert(new_Q.rows() == Q_.rows() && new_Q.cols() == Q_.cols());
    Q_ = new_Q;
    postQUpdate();
}

void MeasurementModel::setTf(const Eigen::Isometry3d& tf) {
    tf_ = tf;
}

uint32_t MeasurementModel::stateSize() const {
    return state_dims_;
}

uint32_t MeasurementModel::measurementSize() const {
    return meas_dims_;
}

}  // namespace state_estimation
