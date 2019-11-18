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
        sigma_pts.col(i + 1) = filter_state_.x + sigma_offset.col(i);
        model->update(sigma_pts.col(i + 1));
        observed_sigma_pts.col(i + 1) = model->h();

        sigma_pts.col(i + 1 + model->measurementSize()) = filter_state_.x - sigma_offset.col(i);
        model->update(sigma_pts.col(i + 1 + model->measurementSize()));
        observed_sigma_pts.col(i + 1 + model->measurementSize()) = model->h();
    }

    // Compute the weighted mean for the predicted measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(model->measurementSize());
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        z_pred += w_mean_(i) * observed_sigma_pts.col(i);
    }

    // Compute the gain
    Eigen::MatrixXd S = model->Q();
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dz = (observed_sigma_pts.col(i) - z_pred);
        S += w_cov_(i) * dz * dz.transpose();
    }

    Eigen::MatrixXd cross_covariance =
        Eigen::MatrixXd::Zero(model->measurementSize(), model->measurementSize());
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        cross_covariance += w_cov_(i) * (sigma_pts.col(i) - sigma_pts.col(0)) *
                            (observed_sigma_pts.col(i) - z_pred).transpose();
    }

    const Eigen::MatrixXd K = cross_covariance * S.inverse();

    // Perform the mean and covariance updates
    filter_state_.x += K * (z - z_pred);
    filter_state_.covariance -= K * S * K.transpose();

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "UKF measurement update:" << std::endl
              << "Sigma offsets=" << std::endl
              << sigma_offset << std::endl
              << "Sigma points=" << std::endl
              << sigma_pts << std::endl
              << "Observed sigma points=" << std::endl
              << observed_sigma_pts << std::endl
              << "z_pred=" << z_pred.transpose() << std::endl
              << "Q=" << std::endl
              << model->Q() << std::endl
              << "S=" << std::endl
              << S << std::endl
              << "Cross Covariance=" << std::endl
              << cross_covariance << std::endl
              << "K=" << std::endl
              << K << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
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
        updateSystemModel(filter_state_.x + sigma_offset.col(i), dt, control, u);
        sigma_pts.col(i + 1) = system_model_->g();

        updateSystemModel(filter_state_.x - sigma_offset.col(i), dt, control, u);
        sigma_pts.col(i + 1 + system_model_->stateSize()) = system_model_->g();
    }

    // Compute the weighted mean
    filter_state_.x.setZero();
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        filter_state_.x += w_mean_(i) * sigma_pts.col(i);
    }

    // Compute the weighted covariance
    filter_state_.covariance = system_model_->R();
    for (uint32_t i = 0; i < num_sigma_pts_; ++i) {
        const Eigen::VectorXd dx = (sigma_pts.col(i) - filter_state_.x);
        filter_state_.covariance += w_cov_(i) * dx * dx.transpose();
    }

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "UKF predicition update:" << std::endl
              << "Sigma offsets=" << std::endl
              << sigma_offset << std::endl
              << "Sigma points=" << std::endl
              << sigma_pts << std::endl
              << "R=" << std::endl
              << system_model_->R() << std::endl
              << "x=" << filter_state_.x.transpose() << std::endl
              << "Covariance=" << std::endl
              << filter_state_.covariance << std::endl;
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
