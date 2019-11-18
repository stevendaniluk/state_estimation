#pragma once

#include <state_estimation/measurement_models/linear_measurement_model.h>
#include <state_estimation/system_models/linear_system_model.h>
#include "filter_base.h"

namespace state_estimation {

// KalmanFilter
//
// Implements an Kalman Filter with linear dynamics.
//
// This class simply provides the update equations for the prediction and correction steps.
class KalmanFilter : public FilterBase<system_models::LinearSystemModel,
                                       measurement_models::LinearMeasurementModel> {
  public:
    using FilterBase::FilterBase;

  protected:
    // myPredict
    //
    // Provides the implementation of the prediction step for a Kalman filter.
    void myPredict(double dt) override;
    void myPredict(const Eigen::VectorXd& u, double dt) override;

    // myPredict
    //
    // Provides the implementation of the correction step for a Kalman filter.
    void myCorrect(const Eigen::VectorXd& z,
                   measurement_models::LinearMeasurementModel* model) override;

  private:
    // KFPredictionUpdate
    //
    // Helper to update the state and covariance via the KF update equations.
    //
    // @param dt: Time delta
    // @param control: When true, a control will be processed
    // @param u: Control vector
    void KFPredictionUpdate(double dt, bool control, Eigen::VectorXd u = Eigen::VectorXd());
};

}  // namespace state_estimation
