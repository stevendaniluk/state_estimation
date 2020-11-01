#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/utilities/angle_utilities.h>

namespace state_estimation {
namespace six_d_rates {

Eigen::VectorXd addState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                         const std::vector<uint8_t>& usage) {
    Eigen::VectorXd result = lhs + rhs;
    result.segment(state::GX, 3) = 9.8062 * result.segment(state::GX, 3).normalized();
    return result;
}

Eigen::VectorXd subtractState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                              const std::vector<uint8_t>& usage) {
    Eigen::VectorXd result = lhs - rhs;
    result.segment(state::GX, 3) = 9.8062 * result.segment(state::GX, 3).normalized();
    return result;
}

Eigen::VectorXd weightedSumOfStates(const Eigen::VectorXd& w, const Eigen::MatrixXd& X,
                                    const std::vector<uint8_t>& usage) {
    assert(w.size() == X.cols());

    Eigen::VectorXd X_sum = Eigen::VectorXd::Zero(X.rows());
    for (size_t i = 0; i < w.size(); ++i) {
        Eigen::VectorXd delta = w(i) * X.col(i);
        delta.segment(state::GX, 3) = 9.8062 * delta.segment(state::GX, 3).normalized();
        X_sum += delta;
    }

    return X_sum;
}

}  // namespace six_d_rates
}  // namespace state_estimation
