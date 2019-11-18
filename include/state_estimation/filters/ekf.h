#pragma once

#include <state_estimation/filters/filter_base.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {

// EKF
//
// Implements an Extended Kalman Filter.
//
// This class simply provides the update equations for the prediction and correction steps in an
// EKF.
class EKF : public FilterBase<system_models::NonlinearSystemModel,
                              measurement_models::NonlinearMeasurementModel> {
  public:
    using FilterBase::FilterBase;

  protected:
    // myPredict
    //
    // Provides the implementation of the prediction step for an EKF.
    void myPredict(double dt) override;
    void myPredict(const Eigen::VectorXd& u, double dt) override;

    // myPredict
    //
    // Provides the implementation of the correction step for an EKF.
    void myCorrect(const Eigen::VectorXd& z,
                   measurement_models::NonlinearMeasurementModel* model) override;

  private:
    // EKFPredictionUpdate
    //
    // Helper to update the state and covariance via the EKF update equations.
    void EKFPredictionUpdate();
};

}  // namespace state_estimation
