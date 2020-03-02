#include <state_estimation/filters/kalman_filter.h>
#include <state_estimation/utilities/data_subset_utilities.h>
#include <Eigen/Dense>

namespace state_estimation {

void KalmanFilter::myPredict(const Eigen::VectorXd& u, double dt) {
    system_model_->update(filter_state_.x, u, dt);

    // Get our sub matrices and vectors
    const Eigen::VectorXd x_subset = getSubset(filter_state_.x, system_model_->activeStates());
    const Eigen::VectorXd u_subset = getSubset(u, system_model_->activeControls());
    const Eigen::MatrixXd cov_subset =
        getSubset(filter_state_.covariance, system_model_->activeStates());
    const Eigen::MatrixXd A_subset = getSubset(system_model_->A(), system_model_->activeStates());
    const Eigen::MatrixXd B_subset = getSubset(system_model_->B(), system_model_->activeStates(),
                                               system_model_->activeControls());
    const Eigen::MatrixXd Rc_subset =
        getSubset(system_model_->Rc(), system_model_->activeControls());
    const Eigen::MatrixXd P_subset =
        getSubset(system_model_->P(), system_model_->activeStates(), {});
    const Eigen::MatrixXd V_subset = getSubset(system_model_->V(), system_model_->activeStates(),
                                               system_model_->activeControls());

    // Update the state
    const Eigen::VectorXd Ax_subset = A_subset * x_subset;
    const Eigen::VectorXd Bu_subset = B_subset * u_subset;
    const Eigen::VectorXd Ax_full = convertSubsetToFullZeroed(
        Ax_subset, system_model_->activeStates(), system_model_->stateSize());
    const Eigen::VectorXd Bu_full = convertSubsetToFullZeroed(
        Bu_subset, system_model_->activeStates(), system_model_->stateSize());

    filter_state_.x = system_model_->addVectors(Ax_full, Bu_full);

    // Update the covariance
    const Eigen::MatrixXd cov_prime_subset = A_subset * cov_subset * A_subset.transpose() +
                                             P_subset * system_model_->Rp() * P_subset.transpose() +
                                             V_subset * Rc_subset * V_subset.transpose();

    convertSubsetToFull(cov_prime_subset, &filter_state_.covariance, system_model_->activeStates());

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "KF predicition update:" << std::endl
              << "A=" << std::endl
              << printMatrix(system_model_->A()) << std::endl
              << "B=" << std::endl
              << printMatrix(system_model_->B()) << std::endl
              << "P=" << std::endl
              << printMatrix(system_model_->P()) << std::endl
              << "V=" << std::endl
              << printMatrix(system_model_->V()) << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

void KalmanFilter::myCorrect(const Eigen::VectorXd& z,
                             measurement_models::LinearMeasurementModel* model, double dt) {
    // Update our measurement model
    model->update(filter_state_.x, dt);

    // Get our sub matrices/vectors
    const Eigen::VectorXd x_subset = getSubset(filter_state_.x, system_model_->activeStates());
    const Eigen::MatrixXd cov_subset =
        getSubset(filter_state_.covariance, system_model_->activeStates());
    const Eigen::MatrixXd C_subset =
        getSubset(model->C(), model->activeMeasurements(), system_model_->activeStates());
    const Eigen::MatrixXd meas_cov_subset =
        getSubset(model->covariance(), model->activeMeasurements());

    // Compute the Kalman gain
    const Eigen::MatrixXd cov_C_T = cov_subset * C_subset.transpose();
    const Eigen::MatrixXd K = cov_C_T * (C_subset * cov_C_T + meas_cov_subset).inverse();

    // Update the state
    const Eigen::VectorXd z_prime_subset = C_subset * x_subset;
    const Eigen::VectorXd z_prime_full = convertSubsetToFullZeroed(
        z_prime_subset, model->activeMeasurements(), model->measurementSize());
    const Eigen::VectorXd dz_full = model->subtractVectors(z, z_prime_full);
    const Eigen::VectorXd dz_subset = getSubset(dz_full, model->activeMeasurements());
    const Eigen::VectorXd dx_subset = K * dz_subset;
    const Eigen::VectorXd dx_full = convertSubsetToFullZeroed(
        dx_subset, system_model_->activeStates(), system_model_->stateSize());

    filter_state_.x = system_model_->addVectors(filter_state_.x, dx_full);

    // Update the covariance
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cov_subset.rows(), cov_subset.rows());
    const Eigen::MatrixXd cov_prime = (I - K * C_subset) * cov_subset;

    convertSubsetToFull(cov_prime, &filter_state_.covariance, system_model_->activeStates());

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "KF measurement update:" << std::endl
              << "C=" << std::endl
              << printMatrix(model->C()) << std::endl
              << "z'=" << printMatrix(z_prime_full) << std::endl
              << "Q=" << std::endl
              << printMatrix(model->covariance()) << std::endl
              << "K=" << std::endl
              << printMatrix(K) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

}  // namespace state_estimation
