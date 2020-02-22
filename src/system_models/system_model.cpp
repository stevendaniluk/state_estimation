#include <assert.h>
#include <state_estimation/system_models/system_model.h>

namespace state_estimation {

SystemModel::SystemModel(uint32_t n, uint32_t m)
    : FilterModel::FilterModel(n)
    , control_dims_(m)
    , R_p_(Eigen::MatrixXd::Zero(n, n))
    , R_c_(Eigen::MatrixXd::Zero(m, m))
    , P_(Eigen::MatrixXd::Zero(n, n))
    , V_(Eigen::MatrixXd::Zero(n, m)) {
    cov_ = Eigen::MatrixXd::Zero(n, n);
}

uint32_t SystemModel::controlSize() const {
    return control_dims_;
}

void SystemModel::update(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) {
    assert(x.size() == state_dims_);
    assert(u.size() == control_dims_);
    myUpdate(x, u, dt);
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
