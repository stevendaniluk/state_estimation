#include <state_estimation/filters/ekf.h>
#include <state_estimation/utilities/logging.h>
#include <Eigen/Dense>

namespace state_estimation {

void EKF::myPredict(const Eigen::VectorXd& u, double dt) {
    system_model_->update(filter_state_.x, u, dt);

    // Update the state and covariance
    filter_state_.x = system_model_->g();
    filter_state_.covariance =
        system_model_->G() * filter_state_.covariance * system_model_->G().transpose() +
        system_model_->P() * system_model_->Rp() * system_model_->P().transpose() +
        system_model_->V() * system_model_->Rc() * system_model_->V().transpose();

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "EKF predicition update:" << std::endl
              << "g=" << printMatrix(system_model_->g()) << std::endl
              << "G=" << std::endl
              << printMatrix(system_model_->G()) << std::endl
              << "P=" << std::endl
              << printMatrix(system_model_->P()) << std::endl
              << "V=" << std::endl
              << printMatrix(system_model_->V()) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

void EKF::myCorrect(const Eigen::VectorXd& z, measurement_models::NonlinearMeasurementModel* model,
                    double dt) {
    // Update our measurement model
    model->update(filter_state_.x, dt);

    // Compute the Kalman gain
    const Eigen::MatrixXd cov_H_T = filter_state_.covariance * model->H().transpose();
    const Eigen::MatrixXd K = cov_H_T * (model->H() * cov_H_T + model->covariance()).inverse();

    // Update the state and covariance with the measurement
    const Eigen::MatrixXd I =
        Eigen::MatrixXd::Identity(filter_state_.x.rows(), filter_state_.x.rows());
    const Eigen::VectorXd dx = K * model->subtractVectors(z, model->h());
    filter_state_.x = system_model_->addVectors(filter_state_.x, dx);
    filter_state_.covariance = (I - K * model->H()) * filter_state_.covariance;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "EKF measurement update:" << std::endl
              << "h=" << printMatrix(model->h()) << std::endl
              << "H=" << std::endl
              << printMatrix(model->H()) << std::endl
              << "Q=" << std::endl
              << printMatrix(model->covariance()) << std::endl
              << "K=" << std::endl
              << printMatrix(K) << std::endl
              << "Innovation=" << printMatrix(dx) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

}  // namespace state_estimation
