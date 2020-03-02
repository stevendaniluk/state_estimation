#include <assert.h>
#include <state_estimation/system_models/linear_system_model.h>

namespace state_estimation {
namespace system_models {

LinearSystemModel::LinearSystemModel(uint16_t n, uint16_t m, uint16_t p)
    : SystemModel::SystemModel(n, m, p)
    , A_(Eigen::MatrixXd::Zero(state_dims_, state_dims_))
    , B_(Eigen::MatrixXd::Zero(state_dims_, control_dims_)) {}

Eigen::MatrixXd LinearSystemModel::A() const {
    return A_;
}

void LinearSystemModel::setA(const Eigen::MatrixXd& new_A) {
    assert(new_A.rows() == state_dims_ && new_A.cols() == state_dims_);
    A_ = new_A;
}

Eigen::MatrixXd LinearSystemModel::B() const {
    return B_;
}

void LinearSystemModel::setB(const Eigen::MatrixXd& new_B) {
    assert(new_B.rows() == state_dims_ && new_B.cols() == control_dims_);
    B_ = new_B;
}

}  // namespace system_models
}  // namespace state_estimation
