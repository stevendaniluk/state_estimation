#include <assert.h>
#include <state_estimation/system_models/system_model.h>

namespace state_estimation {

SystemModel::SystemModel(uint16_t n, uint16_t m, uint16_t p)
    : FilterModel::FilterModel(n)
    , control_dims_(m)
    , control_usage_(m, 0)
    , R_p_(Eigen::MatrixXd::Zero(p, p))
    , R_c_(Eigen::MatrixXd::Zero(m, m))
    , P_(Eigen::MatrixXd::Zero(state_dims_, p))
    , V_(Eigen::MatrixXd::Zero(state_dims_, m)) {
    setActiveControls({});
}

void SystemModel::update(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) {
    assert(x.size() == state_dims_);
    assert(u.size() == control_dims_);
    myUpdate(x, u, dt);
}

uint16_t SystemModel::controlSize() const {
    return control_dims_;
}

uint16_t SystemModel::activeControlSize() const {
    return active_controls_.empty() ? control_dims_ : active_controls_.size();
}

std::vector<uint16_t> SystemModel::activeControls() const {
    return active_controls_;
}

std::vector<uint8_t> SystemModel::controlUsage() const {
    return control_usage_;
}

void SystemModel::setActiveControls(const std::vector<uint16_t>& active_controls) {
    // Only grab the valid indices
    active_controls_.clear();
    for (auto index : active_controls) {
        if (index < control_dims_) {
            active_controls_.push_back(index);
        }
    }

    // Mark which controls are used
    if (!active_controls_.empty()) {
        std::fill(control_usage_.begin(), control_usage_.end(), 0);
        for (auto index : active_controls_) {
            control_usage_[index] = 1;
        }
    } else {
        // All controls are used
        std::fill(control_usage_.begin(), control_usage_.end(), 1);
    }

    // Converting to/from subsets assumes an ordered list of indices
    std::sort(active_controls_.begin(), active_controls_.end());
}

void SystemModel::setProcessCovariance(const Eigen::MatrixXd& R_p) {
    assert(R_p.rows() == R_p_.rows() && R_p.cols() == R_p_.cols());
    R_p_ = R_p;
}

void SystemModel::setControlCovariance(const Eigen::MatrixXd& R_c) {
    assert(R_c.rows() == control_dims_ && R_c.cols() == control_dims_);
    R_c_ = R_c;
}

Eigen::MatrixXd SystemModel::Rp() const {
    return R_p_;
}

Eigen::MatrixXd SystemModel::Rc() const {
    return R_c_;
}

Eigen::MatrixXd SystemModel::P() const {
    return P_;
}

Eigen::MatrixXd SystemModel::V() const {
    return V_;
}

}  // namespace state_estimation
