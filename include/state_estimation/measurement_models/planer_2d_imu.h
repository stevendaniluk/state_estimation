#pragma once

#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

// Planer2DImu
//
// Defines a measurement model for an IMU with a Planer 2D system.
//
// This operates on a Planer2D state vector and a a 6 DOF IMU measurement.
class Planer2DImu : public NonlinearMeasurementModel {
  public:
    // Constructor
    //
    // @param compute_covariance: When true the covariance will be computed on each update
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    // @param include_gravity: When true gravity will be included in the predicted measurement and
    //                         measurement Jacobian
    Planer2DImu(bool compute_covariance, bool compute_jacobian, bool include_gravity = true,
                double gravity = 9.80665);

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override;

    // Flag for if gravity should be included in the measurement
    bool include_gravity_;
    // Gravitational acceleration to use
    double gravity_;
};

}  // namespace measurement_models
}  // namespace state_estimation
