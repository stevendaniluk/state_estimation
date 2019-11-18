#include <state_estimation/filters/ekf.h>
#include <Eigen/Dense>

namespace state_estimation {

void EKF::myPredict(double dt) {
    system_model_->updateNoControl(filter_state_.x, dt);
    EKFPredictionUpdate();
}

void EKF::myPredict(const Eigen::VectorXd& u, double dt) {
    system_model_->update(filter_state_.x, u, dt);
    EKFPredictionUpdate();
}

void EKF::myCorrect(const Eigen::VectorXd& z,
                    measurement_models::NonlinearMeasurementModel* model) {
    // Update our measurement model
    model->update(filter_state_.x);

    // Compute the Kalman gain
    const Eigen::MatrixXd cov_H_T = filter_state_.covariance * model->H().transpose();
    const Eigen::MatrixXd K = cov_H_T * (model->H() * cov_H_T + model->Q()).inverse();

    // Update the state and covariance with the measurement
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(filter_state_.x.rows(), filter_state_.x.rows());
    filter_state_.x += K * (z - model->h());
    filter_state_.covariance = (I - K * model->H()) * filter_state_.covariance;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "EKF measurement update:" << std::endl
              << "h=[" << model->h().transpose() << "]" << std::endl
              << "H=" << std::endl
              << model->H() << std::endl
              << "Q=" << std::endl
              << model->Q() << std::endl
              << "K=" << std::endl
              << K << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
#endif
}

void EKF::EKFPredictionUpdate() {
    // Update the state and covariance
    filter_state_.x = system_model_->g();
    filter_state_.covariance =
        system_model_->G() * filter_state_.covariance * system_model_->G().transpose() +
        system_model_->R();

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "EKF predicition update:" << std::endl
              << "g=[" << system_model_->g().transpose() << "]" << std::endl
              << "G=" << std::endl
              << system_model_->G() << std::endl
              << "R=" << std::endl
              << system_model_->R() << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
#endif
}

}  // namespace state_estimation
