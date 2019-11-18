#include <state_estimation/filters/kalman_filter.h>
#include <Eigen/Dense>

namespace state_estimation {

void KalmanFilter::myPredict(double dt) {
    KFPredictionUpdate(dt, false);
}

void KalmanFilter::myPredict(const Eigen::VectorXd& u, double dt) {
    KFPredictionUpdate(dt, true, u);
}

void KalmanFilter::myCorrect(const Eigen::VectorXd& z,
                             measurement_models::LinearMeasurementModel* model) {
    // Update our measurement model
    model->update(filter_state_.x);

    // Compute the Kalman gain
    const Eigen::MatrixXd cov_C_T = filter_state_.covariance * model->C().transpose();
    const Eigen::MatrixXd K = cov_C_T * (model->C() * cov_C_T + model->Q()).inverse();

    // Update the state and covariance with the measurement
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(filter_state_.x.rows(), filter_state_.x.rows());
    filter_state_.x += K * (z - model->C() * filter_state_.x);
    filter_state_.covariance = (I - K * model->C()) * filter_state_.covariance;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "KF measurement update:" << std::endl
              << "C=" << std::endl
              << model->C() << std::endl
              << "Q=" << std::endl
              << model->Q() << std::endl
              << "K=" << std::endl
              << K << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
#endif
}

void KalmanFilter::KFPredictionUpdate(double dt, bool control, Eigen::VectorXd u) {
    if (!control) {
        system_model_->updateNoControl(filter_state_.x, dt);
        filter_state_.x = system_model_->A() * filter_state_.x;
    } else {
        system_model_->update(filter_state_.x, u, dt);
        filter_state_.x = system_model_->A() * filter_state_.x + system_model_->B() * u;
    }

    filter_state_.covariance =
        system_model_->A() * filter_state_.covariance * system_model_->A().transpose() +
        system_model_->R();

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "KF predicition update:" << std::endl
              << "A=" << std::endl
              << system_model_->A() << std::endl
              << "B=" << std::endl
              << system_model_->B() << std::endl
              << "R=" << std::endl
              << system_model_->R() << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
#endif
}

}  // namespace state_estimation
