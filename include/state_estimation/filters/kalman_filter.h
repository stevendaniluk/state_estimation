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
    void myPredict(const Eigen::VectorXd& u, double dt) override;

    // myPredict
    //
    // Provides the implementation of the correction step for a Kalman filter.
    void myCorrect(const Eigen::VectorXd& z, measurement_models::LinearMeasurementModel* model,
                   double dt) override;
};

}  // namespace state_estimation
