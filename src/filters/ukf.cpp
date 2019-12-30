#include <state_estimation/filters/ukf.h>
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
    uint32_t n = system_model_->g().size();
    num_sigma_pts_ = 2 * system_model_->stateSize() + 1;

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

void UKF::myPredict(double dt) {
    UKFPredictionUpdate(dt, false);
}

void UKF::myPredict(const Eigen::VectorXd& u, double dt) {
    UKFPredictionUpdate(dt, true, u);
}

void UKF::myCorrect(const Eigen::VectorXd& z,
                    measurement_models::NonlinearMeasurementModel* model) {
    // Generate the sigma points and run them through the measurement model
    Eigen::MatrixXd sigma_offset =
        ((system_model_->stateSize() + lambda_) * filter_state_.covariance).llt().matrixL();
    Eigen::MatrixXd sigma_pts(model->measurementSize(), num_sigma_pts_);
    Eigen::MatrixXd observed_sigma_pts(model->measurementSize(), num_sigma_pts_);

    sigma_pts.col(0) = filter_state_.x;
    model->update(filter_state_.x);
    observed_sigma_pts.col(0) = model->h();

    for (uint32_t i = 0; i < system_model_->stateSize(); ++i) {
        sigma_pts.col(i + 1) = system_model_->addVectors(filter_state_.x, sigma_offset.col(i));
        model->update(sigma_pts.col(i + 1));
        observed_sigma_pts.col(i + 1) = model->h();

        sigma_pts.col(i + 1 + model->measurementSize()) =
            system_model_->subtractVectors(filter_state_.x, sigma_offset.col(i));
        model->update(sigma_pts.col(i + 1 + model->measurementSize()));
        observed_sigma_pts.col(i + 1 + model->measurementSize()) = model->h();
    }

    // Compute the weighted mean for the predicted measurement
    Eigen::VectorXd z_pred = model->weightedSum(w_mean_, observed_sigma_pts);

    // Compute the gain
    Eigen::MatrixXd S = model->covariance();
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dz = model->subtractVectors(observed_sigma_pts.col(i), z_pred);
        S += w_cov_(i) * dz * dz.transpose();
    }

    Eigen::MatrixXd cross_covariance =
        Eigen::MatrixXd::Zero(model->measurementSize(), model->measurementSize());
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dx =
            system_model_->subtractVectors(sigma_pts.col(i), sigma_pts.col(0));
        const Eigen::VectorXd dz = model->subtractVectors(observed_sigma_pts.col(i), z_pred);

        cross_covariance += w_cov_(i) * dx * dz.transpose();
    }

    const Eigen::MatrixXd K = cross_covariance * S.inverse();

    // Perform the mean and covariance updates
    const Eigen::VectorXd dx = K * model->subtractVectors(z, z_pred);
    filter_state_.x = system_model_->addVectors(filter_state_.x, dx);
    filter_state_.covariance -= K * S * K.transpose();

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "UKF measurement update:" << std::endl
              << "Sigma offsets=" << std::endl
              << printMatrix(sigma_offset) << std::endl
              << "Sigma points=" << std::endl
              << printMatrix(sigma_pts) << std::endl
              << "Observed sigma points=" << std::endl
              << printMatrix(observed_sigma_pts) << std::endl
              << "z_pred=" << printMatrix(z_pred) << std::endl
              << "Q=" << std::endl
              << printMatrix(model->covariance()) << std::endl
              << "S=" << std::endl
              << printMatrix(S) << std::endl
              << "Cross Covariance=" << std::endl
              << printMatrix(cross_covariance) << std::endl
              << "K=" << std::endl
              << printMatrix(K) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

void UKF::UKFPredictionUpdate(double dt, bool control, Eigen::VectorXd u) {
    // Generate the sigma points and run them through the system model
    const Eigen::MatrixXd sigma_offset =
        ((system_model_->stateSize() + lambda_) * filter_state_.covariance).llt().matrixL();
    Eigen::MatrixXd sigma_pts(system_model_->stateSize(), num_sigma_pts_);

    updateSystemModel(filter_state_.x, dt, control, u);
    Eigen::VectorXd x = system_model_->g();
    sigma_pts.col(0) = system_model_->g();

    for (uint32_t i = 0; i < system_model_->stateSize(); ++i) {
        const Eigen::VectorXd x_high =
            system_model_->addVectors(filter_state_.x, sigma_offset.col(i));
        updateSystemModel(x_high, dt, control, u);
        sigma_pts.col(i + 1) = system_model_->g();

        const Eigen::VectorXd x_low =
            system_model_->subtractVectors(filter_state_.x, sigma_offset.col(i));
        updateSystemModel(x_low, dt, control, u);
        sigma_pts.col(i + 1 + system_model_->stateSize()) = system_model_->g();
    }

    // Compute the weighted mean
    filter_state_.x = system_model_->weightedSum(w_mean_, sigma_pts);

    // Compute the weighted covariance
    filter_state_.covariance = system_model_->covariance();
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dx =
            system_model_->subtractVectors(sigma_pts.col(i), filter_state_.x);
        filter_state_.covariance += w_cov_(i) * dx * dx.transpose();
    }

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "UKF predicition update:" << std::endl
              << "Sigma offsets=" << std::endl
              << printMatrix(sigma_offset) << std::endl
              << "Sigma points=" << std::endl
              << printMatrix(sigma_pts) << std::endl
              << "R=" << std::endl
              << printMatrix(system_model_->covariance()) << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "Covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

void UKF::updateSystemModel(const Eigen::VectorXd& x, double dt, bool control, Eigen::VectorXd u) {
    if (!control) {
        system_model_->updateNoControl(x, dt);
    } else {
        system_model_->update(x, u, dt);
    }
}

}  // namespace state_estimation
