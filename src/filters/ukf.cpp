#include <state_estimation/filters/ukf.h>
#include <state_estimation/utilities/data_subset_utilities.h>
#include <Eigen/Dense>

namespace state_estimation {

UKF::UKF(system_models::NonlinearSystemModel* system_model)
    : FilterBase::FilterBase(system_model) {
    initializeSigmaPointParameters();
}

UKF::UKF(system_models::NonlinearSystemModel* system_model, const Eigen::VectorXd& x,
         const Eigen::MatrixXd& cov, double timestamp)
    : FilterBase::FilterBase(system_model, x, cov, timestamp) {
    initializeSigmaPointParameters();
}

void UKF::setSigmaPointParameters(double alpha, double kappa, double beta) {
    uint32_t n = system_model_->activeStateSize();
    num_sigma_pts_ = 2 * n + 1;

    // Compute our lambda value
    lambda_ = pow(alpha, 2) * (n + kappa) - n;

    // Compute our weight vectors
    w_mean_.resize(num_sigma_pts_);
    w_cov_.resize(num_sigma_pts_);

    const double init_w = 0.5 / (n + lambda_);
    w_mean_ = Eigen::VectorXd::Constant(num_sigma_pts_, init_w);
    w_cov_ = Eigen::VectorXd::Constant(num_sigma_pts_, init_w);

    w_mean_(0) = lambda_ / (n + lambda_);
    w_cov_(0) = w_mean_(0) + 1 - pow(alpha, 2) + beta;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "UKF sigma point initialization" << std::endl
              << "alpha=" << alpha << std::endl
              << "kappa=" << kappa << std::endl
              << "lambda=" << lambda_ << std::endl
              << "Initialized mean weights to [" << w_mean_.transpose() << "]" << std::endl
              << "Initialized covariance weights to [" << w_cov_.transpose() << "]" << std::endl;
#endif
}

void UKF::initializeSigmaPointParameters() {
    setSigmaPointParameters(0.001, 0, 2);
}

void UKF::myPredict(const Eigen::VectorXd& u, double dt) {
    // We will only be updating a subset of the state. All the computationally expensive linear
    // algebra operations will happen on the subset of the state for efficiency. Since the models
    // still operate on the full state vector the sigma points vectors will be of the full
    // dimensionality, but the number of sigma points will still be determined by the number of
    // active states. This only adds a slight additional memory cost.

    // Generate the sigma offsets only using the subset of the covariance matrix we are inerested
    // for efficiency
    const Eigen::MatrixXd cov_subset = getSubset(
        filter_state_.covariance, system_model_->activeStates(), system_model_->activeStates());
    const Eigen::MatrixXd sigma_offset_subset =
        ((system_model_->activeStateSize() + lambda_) * cov_subset).llt().matrixL();

    Eigen::MatrixXd sigma_pts(system_model_->stateSize(), num_sigma_pts_);

    // Run all the sigma points through the model
    system_model_->update(filter_state_.x, u, dt);
    sigma_pts.col(0) = system_model_->g();

    Eigen::VectorXd offset = Eigen::VectorXd::Zero(system_model_->stateSize());
    for (uint32_t i = 0; i < system_model_->activeStateSize(); ++i) {
        const uint32_t i_high = i + 1;
        const uint32_t i_low = i + 1 + system_model_->activeStateSize();
        convertSubsetToFull(sigma_offset_subset.col(i), &offset, system_model_->activeStates());

        const Eigen::VectorXd x_high = system_model_->addVectors(filter_state_.x, offset);
        system_model_->update(x_high, u, dt);
        sigma_pts.col(i_high) = system_model_->g();

        const Eigen::VectorXd x_low = system_model_->subtractVectors(filter_state_.x, offset);
        system_model_->update(x_low, u, dt);
        sigma_pts.col(i_low) = system_model_->g();
    }

    // Compute the weighted mean
    filter_state_.x = system_model_->weightedSum(w_mean_, sigma_pts);

    // Initialze the covariance with the process and control noise
    const Eigen::MatrixXd Rc_subset =
        getSubset(system_model_->Rc(), system_model_->activeControls());
    const Eigen::MatrixXd P_subset =
        getSubset(system_model_->P(), system_model_->activeStates(), {});
    const Eigen::MatrixXd V_subset = getSubset(system_model_->V(), system_model_->activeStates(),
                                               system_model_->activeControls());

    Eigen::MatrixXd cov_prime_subset = P_subset * system_model_->Rp() * P_subset.transpose() +
                                       V_subset * Rc_subset * V_subset.transpose();

    // Add the weighted sample covariance
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dx_full =
            system_model_->subtractVectors(sigma_pts.col(i), filter_state_.x);
        const Eigen::VectorXd dx_subset = getSubset(dx_full, system_model_->activeStates());

        cov_prime_subset += w_cov_(i) * dx_subset * dx_subset.transpose();
    }

    // Update the full covariance matrix with the subset we calculated
    convertSubsetToFull(cov_prime_subset, &filter_state_.covariance, system_model_->activeStates());

#ifdef DEBUG_STATE_ESTIMATION
    Eigen::MatrixXd sigma_offset =
        Eigen::MatrixXd::Zero(system_model_->stateSize(), system_model_->stateSize());
    convertSubsetToFull(sigma_offset_subset, &sigma_offset, system_model_->activeStates());

    std::cout << "UKF predicition update:" << std::endl
              << "Sigma offsets=" << std::endl
              << printMatrix(sigma_offset) << std::endl
              << "Sigma points=" << std::endl
              << printMatrix(sigma_pts) << std::endl
              << "P=" << std::endl
              << printMatrix(system_model_->P()) << std::endl
              << "V=" << std::endl
              << printMatrix(system_model_->V()) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

void UKF::myCorrect(const Eigen::VectorXd& z, measurement_models::NonlinearMeasurementModel* model,
                    double dt) {
    // We will only be updating a subset of the state. All the computationally expensive linear
    // algebra operations will happen on the subset of the state for efficiency. Since the models
    // still operate on the full state vector the sigma points vectors will be of the full
    // dimensionality, but the number of sigma points will still be determined by the number of
    // active states. This only adds a slight additional memory cost.

    // Generate the sigma offsets only using the subset of the covariance matrix we are inerested
    // for efficiency
    const Eigen::MatrixXd cov_subset =
        getSubset(filter_state_.covariance, system_model_->activeStates());
    const Eigen::MatrixXd sigma_offset_subset =
        ((system_model_->activeStateSize() + lambda_) * cov_subset).llt().matrixL();

    Eigen::MatrixXd state_sigma_pts(system_model_->stateSize(), num_sigma_pts_);
    Eigen::MatrixXd meas_sigma_pts(model->measurementSize(), num_sigma_pts_);

    // Run all the sigma points through the model
    state_sigma_pts.col(0) = filter_state_.x;
    model->update(filter_state_.x, dt);
    meas_sigma_pts.col(0) = model->h();

    Eigen::VectorXd offset = Eigen::VectorXd::Zero(system_model_->stateSize());
    for (uint32_t i = 0; i < system_model_->activeStateSize(); ++i) {
        const uint32_t i_high = i + 1;
        const uint32_t i_low = i + 1 + system_model_->activeStateSize();
        convertSubsetToFull(sigma_offset_subset.col(i), &offset, system_model_->activeStates());

        state_sigma_pts.col(i_high) = system_model_->addVectors(filter_state_.x, offset);
        model->update(state_sigma_pts.col(i_high), dt);
        meas_sigma_pts.col(i_high) = model->h();

        state_sigma_pts.col(i_low) = system_model_->subtractVectors(filter_state_.x, offset);
        model->update(state_sigma_pts.col(i_low), dt);
        meas_sigma_pts.col(i_low) = model->h();
    }

    // Compute the weighted mean for the predicted measurement
    Eigen::VectorXd z_pred = model->weightedSum(w_mean_, meas_sigma_pts);

    // Compute the gain (only using the subset of the states)
    Eigen::MatrixXd S = getSubset(model->covariance(), model->activeMeasurements());
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dz_full = model->subtractVectors(meas_sigma_pts.col(i), z_pred);
        const Eigen::VectorXd dz_subset = getSubset(dz_full, model->activeMeasurements());
        S += w_cov_(i) * dz_subset * dz_subset.transpose();
    }

    Eigen::MatrixXd cross_covariance =
        Eigen::MatrixXd::Zero(system_model_->activeStateSize(), model->activeMeasurementSize());
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dx_full =
            system_model_->subtractVectors(state_sigma_pts.col(i), state_sigma_pts.col(0));
        const Eigen::VectorXd dx_subset = getSubset(dx_full, system_model_->activeStates());

        const Eigen::VectorXd dz_full = model->subtractVectors(meas_sigma_pts.col(i), z_pred);
        const Eigen::VectorXd dz_subset = getSubset(dz_full, model->activeMeasurements());

        cross_covariance += w_cov_(i) * dx_subset * dz_subset.transpose();
    }

    const Eigen::MatrixXd K = cross_covariance * S.inverse();

    // Perform the mean update
    const Eigen::VectorXd dz_full = model->subtractVectors(z, z_pred);
    const Eigen::VectorXd dz_subset = getSubset(dz_full, model->activeMeasurements());
    const Eigen::VectorXd dx_subset = K * dz_subset;
    const Eigen::VectorXd dx_full = convertSubsetToFullZeroed(
        dx_subset, system_model_->activeStates(), system_model_->stateSize());
    filter_state_.x = system_model_->addVectors(filter_state_.x, dx_full);

    // Perform the covariance update
    const Eigen::MatrixXd cov_prime_subset = cov_subset - K * S * K.transpose();
    convertSubsetToFull(cov_prime_subset, &filter_state_.covariance, system_model_->activeStates());

#ifdef DEBUG_STATE_ESTIMATION
    const Eigen::MatrixXd sigma_offset = convertSubsetToFullZeroed(
        sigma_offset_subset, system_model_->activeStates(), system_model_->stateSize());

    const Eigen::MatrixXd S_full =
        convertSubsetToFullZeroed(S, model->activeMeasurements(), model->measurementSize());

    const Eigen::MatrixXd cross_full = convertSubsetToFullZeroed(
        cross_covariance, system_model_->activeStates(), model->activeMeasurements(),
        system_model_->stateSize(), model->measurementSize());

    const Eigen::MatrixXd K_full =
        convertSubsetToFullZeroed(K, system_model_->activeStates(), model->activeMeasurements(),
                                  system_model_->stateSize(), model->measurementSize());

    std::cout << "UKF measurement update:" << std::endl
              << "Sigma offsets=" << std::endl
              << printMatrix(sigma_offset) << std::endl
              << "State sigma points=" << std::endl
              << printMatrix(state_sigma_pts) << std::endl
              << "Measurement sigma points=" << std::endl
              << printMatrix(meas_sigma_pts) << std::endl
              << "z_pred=" << printMatrix(z_pred) << std::endl
              << "Q=" << std::endl
              << printMatrix(model->covariance()) << std::endl
              << "S=" << std::endl
              << printMatrix(S_full) << std::endl
              << "Cross Covariance=" << std::endl
              << printMatrix(cross_full) << std::endl
              << "K=" << std::endl
              << printMatrix(K_full) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

}  // namespace state_estimation
