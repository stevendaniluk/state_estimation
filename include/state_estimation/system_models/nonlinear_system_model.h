#pragma once

#include <state_estimation/system_models/system_model.h>
#include <Eigen/Core>

namespace state_estimation {
namespace system_models {

// NonlinearSystemModel
//
// Describes non-linear dynamics of a system for modeling state transitions probabilities for state
// estimation algorithms.
//
// The system is described by:
//   i)   x' = g(x, u): State transition function
//   ii)  G: Jacobian of g(x, u)
//   iii) R: Process covariance (can be time varying)
//
// The update method is responsible for updating all properties of the g(x, u) function and
// Jacobian G. The actual g() and G() methods are just accessors.
class NonlinearSystemModel : public SystemModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param m: Control dimentions
    // @param compute_jacobian: When true the Jacobian, G,  will be computed during the update step
    NonlinearSystemModel(uint32_t n, uint32_t m, bool compute_jacobian = false);

    // g
    //
    // The state transition function, typically denoted as g(x, u). Since the state and control were
    // set in update(), they are not needed as inputs.
    //
    // @return: Estimated state
    Eigen::VectorXd g() const;

    // G
    //
    // @return: Jacobian of the state transition function g about the current state and control
    Eigen::MatrixXd G() const;

  protected:
    // Predicted state
    Eigen::VectorXd x_pred_;
    // State transition Jacobian
    Eigen::MatrixXd G_;
    // Flag for if the Jacobian should be updated
    bool compute_jacobian_;
};

}  // namespace system_models
}  // namespace state_estimation
