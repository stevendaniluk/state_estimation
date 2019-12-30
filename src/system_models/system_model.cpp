#include <assert.h>
#include <state_estimation/system_models/system_model.h>

namespace state_estimation {

SystemModel::SystemModel(uint32_t n, uint32_t m)
    : FilterModel::FilterModel(n)
    , control_dims_(m) {
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

void SystemModel::updateNoControl(const Eigen::VectorXd& x, double dt) {
    assert(x.size() == state_dims_);
    myUpdateNoControl(x, dt);
}

}  // namespace state_estimation
