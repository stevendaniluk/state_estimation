#pragma once

#include <state_estimation/measurement_models/nonlinear_measurement_model.h>

namespace state_estimation {
namespace measurement_models {

// SixDRatesImu
//
// Defines a measurement model for accelerometer and gyroscope measurements from an IMU for a 6DOF
// rates filter.
//
// The measurement is a meas::imu vector. The state frame is expected to being translationally
// coincident with the IMU, but it can be rotated. The linear acceleration and angular velocity
// will be rotated according to the measurement transform, if it was set.
class SixDRatesImu : public NonlinearMeasurementModel {
  public:
    // Constructor
    //
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    SixDRatesImu(bool compute_jacobian = true);

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override;
    void postTfUpdate() override;
};

}  // namespace measurement_models
}  // namespace state_estimation
