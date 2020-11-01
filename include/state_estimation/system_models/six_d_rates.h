#pragma once

#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {
namespace system_models {

// SixDRates
//
// Represents the system dynamics for a 6DOF system that undergoes omni directional motion.
//
// The process noise covariance is the following components in order:
//   -linear jerk
//   -Linear bias jerk
//   -Angular acceleration
//   -Angular bias acceleration
//   -Gravity angular acceleration
class SixDRates : public NonlinearSystemModel {
  public:
    // Constructor
    //
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    // @param update_covariance: When true the process and control covariance Jacobians will be
    //                           updated
    SixDRates(bool compute_jacobian = true, bool update_covariance = true);

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override;

    Eigen::VectorXd addVectors(const Eigen::VectorXd& lhs,
                               const Eigen::VectorXd& rhs) const override;

    Eigen::VectorXd subtractVectors(const Eigen::VectorXd& lhs,
                                    const Eigen::VectorXd& rhs) const override;

    Eigen::VectorXd weightedSum(const Eigen::VectorXd& w, const Eigen::MatrixXd& X) const override;
};

}  // namespace system_models
}  // namespace state_estimation
