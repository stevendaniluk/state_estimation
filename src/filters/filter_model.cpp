#include <assert.h>
#include <state_estimation/filters/filter_model.h>

namespace state_estimation {

FilterModel::FilterModel(uint32_t n)
    : state_dims_(n)
    , cov_(Eigen::MatrixXd::Zero(n, n))
    , check_stationary_(false)
    , tf_(Eigen::Isometry3d::Identity())
    , tf_set_(false) {}

Eigen::MatrixXd FilterModel::covariance() const {
    return cov_;
}

void FilterModel::setCovariance(const Eigen::MatrixXd& new_cov) {
    assert(new_cov.rows() == cov_.rows() && new_cov.cols() == cov_.cols());
    cov_ = new_cov;
}

void FilterModel::setTf(const Eigen::Isometry3d& tf) {
    tf_ = tf;
    tf_set_ = tf_.matrix() != Eigen::Matrix4d::Identity();

    if (tf_set_) {
        postTfUpdate();
    }
}

uint32_t FilterModel::stateSize() const {
    return state_dims_;
}

}  // namespace state_estimation
