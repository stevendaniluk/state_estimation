#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/utilities/angle_utilities.h>

namespace state_estimation {
namespace six_d_rates {

Eigen::VectorXd addState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs, const std::vector<uint8_t>& usage) {
    return lhs + rhs;
}

Eigen::VectorXd subtractState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs, const std::vector<uint8_t>& usage) {
    return lhs - rhs;
}

Eigen::VectorXd weightedSumOfStates(const Eigen::VectorXd& w, const Eigen::MatrixXd& X, const std::vector<uint8_t>& usage) {
    assert(w.size() == X.cols());

    Eigen::VectorXd X_sum = Eigen::VectorXd::Zero(X.rows());
    for (size_t i = 0; i < w.size(); ++i) {
        X_sum += w(i) * X.col(i);
    }

    return X_sum;
}

}// end six_d_rates namespace
} // end state_estimation namespace
