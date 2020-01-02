#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/utilities/angle_utilities.h>

namespace state_estimation {
namespace planer_2d {

Eigen::VectorXd addState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
    Eigen::VectorXd result = lhs + rhs;
    result(state::PSI) = constrainAngle(result(state::PSI));

    return result;
}

Eigen::VectorXd subtractState(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
    Eigen::VectorXd result = lhs - rhs;
    result(state::PSI) = angleDifference(lhs(state::PSI), rhs(state::PSI));

    return result;
}

Eigen::VectorXd weightedSumOfStates(const Eigen::VectorXd& w, const Eigen::MatrixXd& X) {
    assert(w.size() == X.cols());

    Eigen::VectorXd X_sum = Eigen::VectorXd::Zero(X.rows());
    for (size_t i = 0; i < w.size(); ++i) {
        X_sum += w(i) * X.col(i);
    }

    // The heading angle needs to be processed differently to account for angles being circular
    X_sum(state::PSI) = weightedAngleSum(w, X.row(state::PSI));

    return X_sum;
}

}// end planer_2d namespace
} // end state_estimation namespace