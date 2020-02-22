#include <gtest/gtest.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>
#include <Eigen/Core>

// A system model to use for tests. It simply adds the control to the state with:
//   x' = x^n + dt * I + dt * u
//
// The Jacobian is an identity matrix.
//
// The template parameter can add some non linearity to the update step
template <uint32_t N = 1>
class SampleSystemModel : public state_estimation::system_models::NonlinearSystemModel {
  public:
    SampleSystemModel(uint32_t n, uint32_t m)
        : NonlinearSystemModel::NonlinearSystemModel(n, m) {
        setCovariance(1e-3 * Eigen::MatrixXd::Identity(n, n));
        G_ = Eigen::MatrixXd::Identity(n, n);
        R_p_ = 1e-3 * Eigen::MatrixXd::Identity(n, n);
        P_ = Eigen::MatrixXd::Identity(n, n);
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override {
        x_pred_ = x.array().pow(N).matrix() + dt * Eigen::Vector2d::Ones(stateSize()) + dt * u;
    }
};

// Create a measurement model to use for tests that that simply sets the predicted state as the
// measurement vector. The Jacobian will be an identity matrix. This will make it easy to
// introspect the update equations.
//
// The template parameter can add some non linearity to the update step
template <uint32_t N = 1>
class SampleMeasurementModel
    : public state_estimation::measurement_models::NonlinearMeasurementModel {
  public:
    SampleMeasurementModel(uint32_t n, uint32_t k)
        : NonlinearMeasurementModel::NonlinearMeasurementModel(n, k, false, false) {
        H_ = Eigen::MatrixXd::Identity(k, n);
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override {
        z_pred_ = x;
        if (N > 1) {
            Eigen::VectorXd offset = 1e-2 * x.array().pow(N).matrix();
            z_pred_ += offset;
        }
    }
};

// Generic test fixture for testing different types of filters.
//
// The filter type, system type, and measurement model type must all be defined. This sets up a
// filter of the provided type with some initial conditions. It also provides an assortment of
// different input data, as well as vaeriety of tests that can be applied to multiple different
// types of filters.
//
// This uses state, control, and measurement vectors with two dimensions.
//
// Some of the tests assume that the measurement model directly observes the state variables (i.e.
// for a vanilla Kalman filter the C matrix is an identity).
template <typename FilterT, typename SysT, typename MeasT>
class FilterTest : public ::testing::Test {
  protected:
    void SetUp() override { filter.reset(new FilterT(&system_model, x_i, cov_i, t_i)); }

    std::unique_ptr<FilterT> filter;
    SysT system_model = SysT(2, 2);
    MeasT meas_model = MeasT(2, 2);

    // Assortment of input vectors
    Eigen::VectorXd vec_00 = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd vec_11 = Eigen::VectorXd::Constant(2, 1);
    Eigen::VectorXd vec_22 = Eigen::VectorXd::Constant(2, 2);
    Eigen::VectorXd vec_n11 = Eigen::VectorXd::Constant(2, -1);
    Eigen::VectorXd vec_n22 = Eigen::VectorXd::Constant(2, -2);

    // Initial conditions for the filter
    Eigen::VectorXd x_i = vec_00;
    Eigen::MatrixXd cov_i = Eigen::MatrixXd::Identity(2, 2);
    double t_i = 3.14159;

    // Below are an assortment of generic unit tests that can be applied to different types of
    // filters

    void predictUsesSystemModelProcessNoise() {
        double dt = 0.4;

        // Create two filters with two different system models that have different process noise
        // levels
        Eigen::MatrixXd sigma_1 = 1e-2 * Eigen::MatrixXd::Identity(2, 2);
        SysT model_1(2, 2);
        model_1.setProcessCovariance(sigma_1);
        FilterT filter_1(&model_1, x_i, cov_i, t_i);
        filter_1.predict(vec_22, t_i + dt);

        Eigen::MatrixXd sigma_2 = 1e-4 * Eigen::MatrixXd::Identity(2, 2);
        SysT model_2(2, 2);
        model_2.setProcessCovariance(sigma_2);
        FilterT filter_2(&model_2, x_i, cov_i, t_i);
        filter_2.predict(vec_22, t_i + dt);

        // The covariance between the two filters should only differ by the difference between the
        // two process noise levels since the process noise is added on top of the system update
        Eigen::MatrixXd cov_diff = filter_2.getCovariance() - filter_1.getCovariance();
        Eigen::MatrixXd target_diff = sigma_2 - sigma_1;

        EXPECT_TRUE(cov_diff.isApprox(target_diff, 1e-6)) << "Target:" << std::endl
                                                          << target_diff << std::endl
                                                          << "Actual:" << std::endl
                                                          << cov_diff;
    }

    void zeroMeasurementCovarianceProducesZeroEstimateCovariance() {
        // Note: This will only be true for the KF and EKF variants

        // Make our filter have a very large covariance
        filter->initialize(x_i, 10.0 * cov_i, 0.0);

        filter->correct(vec_11, 0.0 * cov_i, 1.0, &meas_model);

        Eigen::MatrixXd target_cov = 0.0 * cov_i;
        EXPECT_TRUE(filter->getCovariance().isApprox(target_cov, 1e-6)) << "Target:" << std::endl
                                                                        << target_cov << std::endl
                                                                        << "Actual:" << std::endl
                                                                        << filter->getCovariance();
    }

    void zeroMeasurementCovarianceSetsStateToPredictedMeasurement() {
        // Note: This will only be true for the KF and EKF variants

        // Make our filter have a very large covariance
        filter->initialize(x_i, 10.0 * cov_i, 0.0);

        // The measurement model has measurements directly observing state variables, so the update
        // state should be the measurement provided
        Eigen::VectorXd meas = vec_22;
        filter->correct(meas, 0.0 * cov_i, 1.0, &meas_model);

        Eigen::VectorXd x_target = meas;
        EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
    }

    void veryLargeMeasurementCovarianceMakesMeasurementIgnored() {
        // Provide a very different measurement, but with a large covariance
        Eigen::MatrixXd cov = 1e12 * cov_i;
        filter->correct(100.0 * vec_11, cov, t_i, &meas_model);

        // Can't use isApprox() for near zero vectors (initial state was 0's), so we'll use the norm
        EXPECT_LT((filter->getState() - x_i).norm(), 1e-3)
            << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();
    }

    void correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement() {
        // Note: This will only be true for the KF and EKF variants

        // Will need to zero the process noise to isolate the correction update
        system_model.setCovariance(Eigen::MatrixXd::Zero(2, 2));

        Eigen::VectorXd meas = vec_22;
        filter->correct(meas, cov_i, t_i, &meas_model);

        Eigen::VectorXd x_target = 0.5 * (x_i + meas);
        EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
    }

    void correctWithEqualCovarianceHalvesTheCovarience() {
        // Note: This will only be true for the KF and EKF variants

        // Will need to zero the process noise to isolate the correction update
        system_model.setCovariance(Eigen::MatrixXd::Zero(2, 2));

        filter->correct(vec_11, cov_i, t_i, &meas_model);

        Eigen::MatrixXd cov_target = 0.5 * cov_i;
        EXPECT_TRUE(filter->getCovariance().isApprox(cov_target, 1e-6)) << "Target:" << std::endl
                                                                        << cov_target << std::endl
                                                                        << "Actual:" << std::endl
                                                                        << filter->getCovariance();
    }
};
