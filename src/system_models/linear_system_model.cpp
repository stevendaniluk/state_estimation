#include <assert.h>
#include <state_estimation/system_models/linear_system_model.h>

namespace state_estimation {
namespace system_models {

LinearSystemModel::LinearSystemModel(uint32_t n, uint32_t m)
    : SystemModel::SystemModel(n, m)
    , A_(Eigen::MatrixXd::Zero(n, n))
    , B_(Eigen::MatrixXd::Zero(n, m)) {}

Eigen::MatrixXd LinearSystemModel::A() const {
    return A_;
}

void LinearSystemModel::setA(const Eigen::MatrixXd& new_A) {
    assert(new_A.rows() == A_.rows() && new_A.cols() == A_.cols());
    A_ = new_A;
}

Eigen::MatrixXd LinearSystemModel::B() const {
    return B_;
}

void LinearSystemModel::setB(const Eigen::MatrixXd& new_B) {
    assert(new_B.rows() == B_.rows() && new_B.cols() == B_.cols());
    B_ = new_B;
}

}  // namespace system_models
}  // namespace state_estimation
