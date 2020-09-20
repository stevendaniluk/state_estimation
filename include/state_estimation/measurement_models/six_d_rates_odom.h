#pragma once

#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

// SixDRatesOdom
//
// Defines a measurement model for wheel odometry for a 6DOF rates filter.
//
// The measurement vector is
class SixDRatesOdom : public NonlinearMeasurementModel {
  public:
    // Constructor
    //
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    SixDRatesOdom(bool compute_jacobian = true);

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override;
    void postTfUpdate() override;
};

}  // namespace measurement_models
}  // namespace state_estimation
