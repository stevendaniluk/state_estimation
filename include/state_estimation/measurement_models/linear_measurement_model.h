#pragma once

#include <state_estimation/measurement_models/measurement_model.h>
#include <Eigen/Core>

namespace state_estimation {
namespace measurement_models {

// LinearMeasurementModel
//
// Describes a linear measurement model for a system, which estimates system states for a given
// measurement.
//
// The measurement model is described by:
//   i) z = Cx: Measurement function which estimates measurements from states
//
// The update method is responsible for updating all time varying matrix elements, if necessary.
class LinearMeasurementModel : public MeasurementModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param k: Measurement dimensions
    LinearMeasurementModel(uint16_t n, uint16_t k);

    // C
    //
    // @return: Observation matrix
    Eigen::MatrixXd C() const;

    // setC
    //
    // @param new_C: New C matrix to use
    void setC(const Eigen::MatrixXd& new_C);

  protected:
    // Observation matrix
    Eigen::MatrixXd C_;
};

}  // namespace measurement_models
}  // namespace state_estimation
