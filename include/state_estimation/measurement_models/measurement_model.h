#pragma once

#include <state_estimation/filters/filter_model.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// MeasurementModel
//
// Base class for defining the interface for measurement models.
//
// Since many operations may be needed for both the measurement prediction and covariance updates,
// this provides an update method so that all computations can be performed once in the same place,
// and the values (e.g. predicted measurement and covariance) can simply be retrieved with
// accessors.
class MeasurementModel : public FilterModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param k: Measurement dimentions
    // @param compute_covariance: When true the covariance will be computed on each update
    MeasurementModel(uint32_t n, uint32_t k, bool compute_covariance);

    // update
    //
    // Updates the measurement model about the current state.
    //
    // @param x: Current system state
    // @param dt: Time difference between the state and the current measurement
    virtual void update(const Eigen::VectorXd& x, double dt);

    // measurementSize
    //
    // @return: Number of measurement variables
    uint32_t measurementSize() const;

  protected:
    // Internal implementation of update() to be defined by derived classes.
    virtual void myUpdate(const Eigen::VectorXd& x, double dt) = 0;

    // Measurement covariance
    Eigen::MatrixXd Q_;
    // Dimension of the measurement vector
    uint32_t meas_dims_;
    // Flag for if the covariance should be updated internally by the model, or set externally
    bool compute_covariance_;
};

}  // namespace state_estimation
