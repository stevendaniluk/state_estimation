#include <state_estimation/filters/ekf_vs.h>
#include <state_estimation/utilities/data_subset_utilities.h>
#include <Eigen/Dense>

namespace state_estimation {

void EKFVS::myPredict(const Eigen::VectorXd& u, double dt) {
    system_model_->update(filter_state_.x, u, dt);

    // Update the state and covariance. For the covariance the update will happen on the state
    // subsets then be converted back to the full dimensionality.
    filter_state_.x = system_model_->g();

    const Eigen::MatrixXd cov_subset =
        getSubset(filter_state_.covariance, system_model_->activeStates());
    const Eigen::MatrixXd G_subset = getSubset(system_model_->G(), system_model_->activeStates());
    const Eigen::MatrixXd Rc_subset =
        getSubset(system_model_->Rc(), system_model_->activeControls());
    const Eigen::MatrixXd P_subset =
        getSubset(system_model_->P(), system_model_->activeStates(), {});
    const Eigen::MatrixXd V_subset = getSubset(system_model_->V(), system_model_->activeStates(),
                                               system_model_->activeControls());

    const Eigen::MatrixXd cov_prime_subset = G_subset * cov_subset * G_subset.transpose() +
                                             P_subset * system_model_->Rp() * P_subset.transpose() +
                                             V_subset * Rc_subset * V_subset.transpose();

    convertSubsetToFull(cov_prime_subset, &filter_state_.covariance, system_model_->activeStates());

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

void EKFVS::myCorrect(const Eigen::VectorXd& z,
                      measurement_models::NonlinearMeasurementModel* model, double dt) {
    // Update our measurement model
    model->update(filter_state_.x, dt);

    // Get our sub matrices/vectors
    const Eigen::MatrixXd cov_subset =
        getSubset(filter_state_.covariance, system_model_->activeStates());
    const Eigen::MatrixXd H_subset =
        getSubset(model->H(), model->activeMeasurements(), system_model_->activeStates());
    const Eigen::MatrixXd meas_cov_subset =
        getSubset(model->covariance(), model->activeMeasurements());

    // Compute the Kalman gain
    const Eigen::MatrixXd cov_H_T = cov_subset * H_subset.transpose();
    const Eigen::MatrixXd K = cov_H_T * (H_subset * cov_H_T + meas_cov_subset).inverse();

    // Update the state
    const Eigen::VectorXd dz_full = model->subtractVectors(z, model->h());
    const Eigen::VectorXd dz_subset = getSubset(dz_full, model->activeMeasurements());
    const Eigen::VectorXd dx_subset = K * dz_subset;
    const Eigen::VectorXd dx_full = convertSubsetToFullZeroed(
        dx_subset, system_model_->activeStates(), system_model_->stateSize());

    filter_state_.x = system_model_->addVectors(filter_state_.x, dx_full);

    // Update the covariance
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cov_subset.rows(), cov_subset.rows());
    const Eigen::MatrixXd cov_prime = (I - K * H_subset) * cov_subset;

    convertSubsetToFull(cov_prime, &filter_state_.covariance, system_model_->activeStates());

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "EKF measurement update:" << std::endl
              << "h=" << printMatrix(model->h()) << std::endl
              << "H=" << std::endl
              << printMatrix(model->H()) << std::endl
              << "Q=" << std::endl
              << printMatrix(model->covariance()) << std::endl
              << "K=" << std::endl
              << printMatrix(K) << std::endl
              << "Innovation=" << printMatrix(dx_full) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

}  // namespace state_estimation
