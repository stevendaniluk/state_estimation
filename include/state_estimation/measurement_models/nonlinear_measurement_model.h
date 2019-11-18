#pragma once

#include <state_estimation/measurement_models/measurement_model.h>
#include <Eigen/Core>

namespace state_estimation {
namespace measurement_models {

// NonlinearMeasurementModel
//
// Describes a non-linear measurement model for a system, which estimates system states for a given
// emasurement.
//
// The measurement model is described by:
//   i)   z = h(x): Measurement function which estimates measurements from states
//   ii)  H: Jacobian of h(x)
//
// The update method is responsible for updating all time varying matrix elements, if necessary.
class NonlinearMeasurementModel : public MeasurementModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param k: Measurement dimentions
    // @param compute_jacobian: When true the Jacobian, G, will be computed during the update step
    NonlinearMeasurementModel(uint32_t n, uint32_t k, bool compute_jacobian = false);

    // h
    //
    // The measurement function, typically denoted as h(x). Since the state was set in update(), it
    // is not needed as an input.
    //
    // @return: Estimated measurement
    Eigen::VectorXd h() const;

    // H
    //
    // @return: Jacobian of the measurement function h about the current state
    Eigen::MatrixXd H() const;

  protected:
    // Predicted measurement
    Eigen::VectorXd z_pred_;
    // Measurement Jacobian
    Eigen::MatrixXd H_;
    // Flag for if the Jacobian should be updated
    bool compute_jacobian_;
};

}  // namespace measurement_models
}  // namespace state_estimation
