#pragma once

#include <state_estimation/filters/filter_base.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>

namespace state_estimation {

// UKF
//
// Implements an Unscented Kalman Filter.
//
// This class simply provides the update equations for the prediction and correction steps in a
// UKF.
//
// This is a fixed state implementation, it will use the entire state, control, and measurement
// vectors (unlike the VS implementation). It will not have the over head of using variable state
// sizes.
class UKF : public FilterBase<system_models::NonlinearSystemModel,
                              measurement_models::NonlinearMeasurementModel> {
  public:
    UKF(system_models::NonlinearSystemModel* system_model);

    UKF(system_models::NonlinearSystemModel* system_model, const Eigen::VectorXd& x,
        const Eigen::MatrixXd& cov, double timestamp);

    // setSigmaPointParameters
    //
    // Defines the sigma point parameters used for the unscented transform.
    //
    // @param alpha: Scaling parameter for sigma points
    // @param kappa: Scaling parameter for sigma points
    // @param beta: Distribution parameter
    void setSigmaPointParameters(double alpha, double kappa = 0, double beta = 2);

  protected:
    // initializeSigmaPointParameters
    //
    // Sets the sigma point parameters to some default values
    void initializeSigmaPointParameters();

    // myPredict
    //
    // Provides the implementation of the prediction step for an UKF.
    void myPredict(const Eigen::VectorXd& u, double dt) override;

    // myPredict
    //
    // Provides the implementation of the correction step for an UKF.
    void myCorrect(const Eigen::VectorXd& z, measurement_models::NonlinearMeasurementModel* model,
                   double dt) override;

    // How many sigma points we have (2n * 1)
    uint32_t num_sigma_pts_;
    // Coefficient used in sigma point offsets
    double lambda_;
    // Weights used for updating the mean estimate
    Eigen::VectorXd w_mean_;
    // Weights used for updating the covariance estimate
    Eigen::VectorXd w_cov_;
};

}  // namespace state_estimation
