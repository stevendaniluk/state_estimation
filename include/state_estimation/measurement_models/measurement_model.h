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
//
// The transform stored by this object represents the transform from the state frame to the
// measurement frame.
class MeasurementModel : public FilterModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param k: Measurement dimensions
    MeasurementModel(uint16_t n, uint16_t k);

    // update
    //
    // Updates the measurement model about the current state.
    //
    // @param x: Current system state
    // @param dt: Time difference between the state and the current measurement
    virtual void update(const Eigen::VectorXd& x, double dt);

    // measurementSize
    //
    // @return: Dimension of the measurement vector
    uint16_t measurementSize() const;

    // activeMeasurementSize
    //
    // @return: Dimensions of the measurement vector being used
    uint16_t activeMeasurementSize() const;

    // activeMeasurements
    //
    // @return: Ordered list of measurement indices being updated
    std::vector<uint16_t> activeMeasurements() const;

    // measurementUsage
    //
    // @return: A bit field of which measurement variables are used (0=not used)
    std::vector<uint8_t> measurementUsage() const;

    // setActiveMeasurements
    //
    // @param active_measurements: Which measurement variables to update (empty updates all)
    void setActiveMeasurements(const std::vector<uint16_t>& active_measurements);

    // covariance
    //
    // @return: Covariance for prediction/correction
    Eigen::MatrixXd covariance() const;

    // setCovariance
    //
    // @param new_cov: New covariance matrix to use
    void setCovariance(const Eigen::MatrixXd& new_cov);

  protected:
    // Internal implementation of update() to be defined by derived classes.
    virtual void myUpdate(const Eigen::VectorXd& x, double dt) = 0;

    // Dimension of the measurement vector
    uint16_t meas_dims_;
    // Which measurement variables are being used (ordered)
    std::vector<uint16_t> active_measurements_;
    // A bit field of which measurement variables are used (0=not used)
    std::vector<uint8_t> measurement_usage_;
    // Measurement covariance
    Eigen::MatrixXd cov_;
};

}  // namespace state_estimation
