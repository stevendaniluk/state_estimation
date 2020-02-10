#pragma once

#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

// Planer2DNhOdometry
//
// Defines a measurement model for wheel odometry with a Planer 2D system with non holonomic
// constrants (e.g. diff drive).
//
// Because the system is non holonomic, we can only observe velocity in the X direction.
// Measurements will directly observes [Vx, VPsi]. As a result, the Jacobian is an identity matrix.
class Planer2DNhOdometry : public NonlinearMeasurementModel {
  public:
    // Constructor
    //
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    Planer2DNhOdometry(bool compute_jacobian = false);

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override;
};

}  // namespace measurement_models
}  // namespace state_estimation
