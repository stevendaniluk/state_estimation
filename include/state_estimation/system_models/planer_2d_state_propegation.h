#pragma once

#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {
namespace system_models {

// Planer2DStatePropegation
//
// Represents the system dynamics for a 2D planer system that has no controls inputs and advances
// the state by simply integrating each state variable.
//
// This updates the states: [X, Y, Vx, Vy, Ax, Ay, Psi, VPsi].
//
// This assumes a constant covariance matrix.
class Planer2DStatePropegation : public NonlinearSystemModel {
  public:
    // Constructor
    //
    // @param compute_jacobian: When true the Jacobian, G,  will be computed during the update step
    Planer2DStatePropegation(bool compute_jacobian = false);

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
