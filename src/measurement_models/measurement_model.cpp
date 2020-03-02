#include <state_estimation/measurement_models/measurement_model.h>
#include <numeric>

namespace state_estimation {

MeasurementModel::MeasurementModel(uint16_t n, uint16_t k)
    : FilterModel::FilterModel(n)
    , meas_dims_(k)
    , measurement_usage_(k, 0)
    , cov_(Eigen::MatrixXd::Zero(k, k)) {
    setActiveMeasurements({});
}

void MeasurementModel::update(const Eigen::VectorXd& x, double dt) {
    assert(x.size() == state_dims_);
    myUpdate(x, dt);
};

uint16_t MeasurementModel::measurementSize() const {
    return meas_dims_;
}

uint16_t MeasurementModel::activeMeasurementSize() const {
    return active_measurements_.empty() ? meas_dims_ : active_measurements_.size();
}

std::vector<uint16_t> MeasurementModel::activeMeasurements() const {
    return active_measurements_;
}

void MeasurementModel::setActiveMeasurements(const std::vector<uint16_t>& active_measurements) {
    // Only grab the valid indices
    active_measurements_.clear();
    for (auto index : active_measurements) {
        if (index < meas_dims_) {
            active_measurements_.push_back(index);
        }
    }

    // Mark which measurements are used
    if (!active_measurements_.empty()) {
        std::fill(measurement_usage_.begin(), measurement_usage_.end(), 0);
        for (auto index : active_measurements_) {
            measurement_usage_[index] = 1;
        }
    } else {
        // All measurements are used
        std::fill(measurement_usage_.begin(), measurement_usage_.end(), 1);
    }

    // Converting to/from subsets assumes an ordered list of indices
    std::sort(active_measurements_.begin(), active_measurements_.end());
}

std::vector<uint8_t> MeasurementModel::measurementUsage() const {
    return measurement_usage_;
}

Eigen::MatrixXd MeasurementModel::covariance() const {
    return cov_;
}

void MeasurementModel::setCovariance(const Eigen::MatrixXd& new_cov) {
    assert(new_cov.rows() == meas_dims_ && new_cov.cols() == meas_dims_);
    cov_ = new_cov;
}

}  // namespace state_estimation
