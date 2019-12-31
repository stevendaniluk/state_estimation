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

void FilterModel::setCheckStationary(bool check) {
    check_stationary_ = check;
}

bool FilterModel::checkStationary() {
    return check_stationary_ && is_stationary_f_ && make_stationary_f_;
}

bool FilterModel::isStationary(const Eigen::VectorXd& x, const Eigen::VectorXd& data) const {
    return is_stationary_f_(x, data);
}

void FilterModel::makeStationary(Eigen::VectorXd* x, Eigen::MatrixXd* cov) const {
    make_stationary_f_(x, cov);
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
