#include <assert.h>
#include <state_estimation/filters/filter_model.h>
#include <algorithm>

namespace state_estimation {

FilterModel::FilterModel(uint16_t n)
    : state_dims_(n)
    , state_usage_(n, 0)
    , tf_(Eigen::Isometry3d::Identity())
    , tf_set_(false) {
    setActiveStates({});
}

void FilterModel::setTf(const Eigen::Isometry3d& tf) {
    tf_ = tf;
    tf_set_ = tf_.matrix() != Eigen::Matrix4d::Identity();

    if (tf_set_) {
        postTfUpdate();
    }
}

uint16_t FilterModel::stateSize() const {
    return state_dims_;
}

uint16_t FilterModel::activeStateSize() const {
    return active_states_.empty() ? state_dims_ : active_states_.size();
}

std::vector<uint16_t> FilterModel::activeStates() const {
    return active_states_;
}

std::vector<uint8_t> FilterModel::stateUsage() const {
    return state_usage_;
}

void FilterModel::setActiveStates(const std::vector<uint16_t>& active_states) {
    // Only grab the valid indices
    active_states_.clear();
    for (auto index : active_states) {
        if (index < state_dims_) {
            active_states_.push_back(index);
        }
    }

    if (!active_states_.empty()) {
        std::fill(state_usage_.begin(), state_usage_.end(), 0);
        for (auto index : active_states_) {
            state_usage_[index] = 1;
        }
    } else {
        // All states are used
        std::fill(state_usage_.begin(), state_usage_.end(), 1);
    }

    // Converting to/from subsets assumes an ordered list of indices
    std::sort(active_states_.begin(), active_states_.end());
}

bool FilterModel::checkStationary() {
    return (bool)is_stationary_f_;
}

bool FilterModel::isStationary(const Eigen::VectorXd& x, const Eigen::VectorXd& data) const {
    return is_stationary_f_(x, data);
}

void FilterModel::makeStationary(Eigen::VectorXd* x, Eigen::MatrixXd* cov) const {
    if (make_stationary_f_) {
        make_stationary_f_(x, cov);
    }
}

void FilterModel::setIsStationaryFunction(
    const std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)>& f) {
    is_stationary_f_ = f;
}

void FilterModel::setMakeStationaryFunction(
    const std::function<void(Eigen::VectorXd*, Eigen::MatrixXd*)>& f) {
    make_stationary_f_ = f;
}

Eigen::VectorXd FilterModel::addVectors(const Eigen::VectorXd& lhs,
                                        const Eigen::VectorXd& rhs) const {
    return lhs + rhs;
}

Eigen::VectorXd FilterModel::subtractVectors(const Eigen::VectorXd& lhs,
                                             const Eigen::VectorXd& rhs) const {
    return lhs - rhs;
}

Eigen::VectorXd FilterModel::weightedSum(const Eigen::VectorXd& w, const Eigen::MatrixXd& X) const {
    assert(w.size() == X.cols());

    Eigen::VectorXd X_sum = Eigen::VectorXd::Zero(X.rows());
    for (size_t i = 0; i < w.size(); ++i) {
        X_sum += w(i) * X.col(i);
    }

    return X_sum;
}

}  // namespace state_estimation
