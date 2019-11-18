#pragma once

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
class MeasurementModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param k: Measurement dimentions
    MeasurementModel(uint32_t n, uint32_t k);

    // update
    //
    // Updates the measurement model about the current state.
    //
    // @param x: Current system state
    virtual void update(const Eigen::VectorXd& x);

    // Q
    //
    // @return Measurement covariance
    Eigen::MatrixXd Q() const;

    // setQ
    //
    // @param new_Q: New measurement covariance to use
    void setQ(const Eigen::MatrixXd& new_Q);

    // setTf
    //
    // @param tf: Transformation from the state frame to the measurement frame
    void setTf(const Eigen::Isometry3d& tf);

    // stateSize
    //
    // @return: Number of state variables
    uint32_t stateSize() const;

    // measurementSize
    //
    // @return: Number of measurement variables
    uint32_t measurementSize() const;

  protected:
    // Internal implementation of update() to be defined by derived classes.
    virtual void myUpdate(const Eigen::VectorXd& x) = 0;

    // Optional operation to perform after updating the measurment covariance
    virtual void postQUpdate() {}

    // Measurement covariance
    Eigen::MatrixXd Q_;
    // Dimension of the state vector
    uint32_t state_dims_;
    // Dimension of the measurement vector
    uint32_t meas_dims_;
    // Transformation from the frame the measurement is taken in, to the state frame
    Eigen::Isometry3d tf_;
};

}  // namespace state_estimation
