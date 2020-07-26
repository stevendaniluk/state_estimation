#include <gtest/gtest.h>
#include <state_estimation/measurement_models/linear_measurement_model.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/linear_system_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>
#include <Eigen/Core>

using namespace state_estimation;

// A system model to use for tests. It simply adds the control to the state with:
//   x' = x^n + dt * I + dt * u
//
// The Jacobian is an identity matrix.
//
// The template parameter can add some non linearity to the update step
template <uint32_t N = 1>
class SampleSystemModel : public system_models::NonlinearSystemModel {
  public:
    SampleSystemModel(uint16_t n, uint16_t m, uint16_t p, bool compute_jacobian = false,
                      bool update_covariance = false)
        : NonlinearSystemModel::NonlinearSystemModel(n, m, p, compute_jacobian, update_covariance) {
        G_ = Eigen::MatrixXd::Identity(state_dims_, state_dims_);
        R_p_ = 1e-3 * Eigen::MatrixXd::Identity(state_dims_, state_dims_);
        P_ = Eigen::MatrixXd::Identity(state_dims_, state_dims_);
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override {
        // Only update the active states with the active controls
        x_pred_ = x;
        for (uint16_t index = 0; index < stateSize(); ++index) {
            if (stateUsage()[index]) {
                x_pred_(index) = pow(x(index), N) + dt;

                if (controlUsage()[index]) {
                    x_pred_(index) += dt * u(index);
                }
            }
        }
    }
};

// Create a measurement model to use for tests that that simply sets the predicted state as the
// measurement vector. The Jacobian will be an identity matrix. This will make it easy to
// introspect the update equations.
//
// The template parameter can add some non linearity to the update step
template <uint32_t N = 1>
class SampleMeasurementModel : public measurement_models::NonlinearMeasurementModel {
  public:
    SampleMeasurementModel(uint16_t n, uint16_t k, bool compute_jacobian = false)
        : NonlinearMeasurementModel::NonlinearMeasurementModel(n, k, compute_jacobian) {
        H_ = Eigen::MatrixXd::Identity(meas_dims_, state_dims_);
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

  public:
    std::unique_ptr<FilterT> filter;
    SysT system_model = SysT(2, 2, 2);
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
        SysT model_1(2, 2, 2);
        model_1.setProcessCovariance(sigma_1);
        FilterT filter_1(&model_1, x_i, cov_i, t_i);
        filter_1.predict(vec_22, t_i + dt);

        Eigen::MatrixXd sigma_2 = 1e-4 * Eigen::MatrixXd::Identity(2, 2);
        SysT model_2(2, 2, 2);
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
        system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));

        Eigen::VectorXd meas = vec_22;
        filter->correct(meas, cov_i, t_i, &meas_model);

        Eigen::VectorXd x_target = 0.5 * (x_i + meas);
        EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
    }

    void correctWithEqualCovarianceHalvesTheCovarience() {
        // Note: This will only be true for the KF and EKF variants

        // Will need to zero the process noise to isolate the correction update
        system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));

        filter->correct(vec_11, cov_i, t_i, &meas_model);

        Eigen::MatrixXd cov_target = 0.5 * cov_i;
        EXPECT_TRUE(filter->getCovariance().isApprox(cov_target, 1e-6)) << "Target:" << std::endl
                                                                        << cov_target << std::endl
                                                                        << "Actual:" << std::endl
                                                                        << filter->getCovariance();
    }

    void predictOnlyUpdatesActiveStates() {
        double dt = 0.4;
        Eigen::VectorXd u = vec_22;

        // Perform a prediction step with the full state for reference
        filter->predict(u, filter->getStateTime() + dt);
        Eigen::VectorXd x_prime_full = filter->getState();
        Eigen::MatrixXd cov_prime_full = filter->getCovariance();

        // Only update one state
        system_model.setActiveStates({1});
        filter.reset(new FilterT(&system_model, x_i, cov_i, t_i));

        filter->predict(u, filter->getStateTime() + dt);

        EXPECT_EQ(x_i(0), filter->getState()(0));
        EXPECT_FLOAT_EQ(x_prime_full(1), filter->getState()(1));

        EXPECT_EQ(cov_i(0, 0), filter->getCovariance()(0, 0));
        EXPECT_FLOAT_EQ(cov_prime_full(1, 1), filter->getCovariance()(1, 1));
    }

    void predictOnlyUsesActiveControls() {
        double dt = 0.4;
        Eigen::VectorXd u = vec_22;

        // Perform a prediction step with the full state for reference
        filter->predict(0 * u, filter->getStateTime() + dt);
        Eigen::VectorXd x_prime_no_control = filter->getState();

        filter->initialize(x_i, cov_i, t_i);
        filter->predict(u, filter->getStateTime() + dt);
        Eigen::VectorXd x_prime_control = filter->getState();

        // Only provide one control input
        system_model.setActiveControls({1});
        filter.reset(new FilterT(&system_model, x_i, cov_i, t_i));

        filter->predict(u, filter->getStateTime() + dt);

        EXPECT_EQ(x_prime_no_control(0), filter->getState()(0));
        EXPECT_FLOAT_EQ(x_prime_control(1), filter->getState()(1));
    }

    void correctOnlyUpdatesActiveStates() {
        Eigen::VectorXd z = vec_22;
        Eigen::MatrixXd cov = 1e-2 * Eigen::MatrixXd::Identity(2, 2);

        // Perform a correction step with the full state for reference
        filter->correct(z, cov, filter->getStateTime(), &meas_model);
        Eigen::VectorXd x_prime_full = filter->getState();
        Eigen::MatrixXd cov_prime_full = filter->getCovariance();

        // Only update one state
        system_model.setActiveStates({1});
        meas_model.setActiveStates({1});
        meas_model.setActiveMeasurements({1});
        filter.reset(new FilterT(&system_model, x_i, cov_i, t_i));

        filter->correct(z, cov, filter->getStateTime(), &meas_model);

        EXPECT_EQ(x_i(0), filter->getState()(0));
        EXPECT_FLOAT_EQ(x_prime_full(1), filter->getState()(1));

        EXPECT_EQ(cov_i(0, 0), filter->getCovariance()(0, 0));
        EXPECT_FLOAT_EQ(cov_prime_full(1, 1), filter->getCovariance()(1, 1));
    }

    void correctOnlyUsesActiveMeasurements() {
        Eigen::VectorXd z = vec_22;
        Eigen::MatrixXd cov = 1e-2 * Eigen::MatrixXd::Identity(2, 2);

        // Perform a correction step with the full state for reference
        filter->correct(z, cov, filter->getStateTime(), &meas_model);
        Eigen::VectorXd x_prime_full = filter->getState();
        Eigen::MatrixXd cov_prime_full = filter->getCovariance();

        // Only utilize one measurement variable
        meas_model.setActiveMeasurements({1});
        filter.reset(new FilterT(&system_model, x_i, cov_i, t_i));

        filter->correct(z, cov, filter->getStateTime(), &meas_model);

        EXPECT_EQ(x_i(0), filter->getState()(0));
        EXPECT_FLOAT_EQ(x_prime_full(1), filter->getState()(1));

        EXPECT_EQ(cov_i(0, 0), filter->getCovariance()(0, 0));
        EXPECT_FLOAT_EQ(cov_prime_full(1, 1), filter->getCovariance()(1, 1));
    }
};

/////////////////////////////////
// Kalman filter test utilities.
//
// Create a system model to use for tests. It simply adds the control to the state with:
//   x' = Ax + dt * u
//
// Create a measurement model to use for tests that that simply sets the predicted state as the
// measurement vector. The C matrix will be an identity matrix. This will make it easy to
// introspect the update equations.

class KFSampleSystemModel : public system_models::LinearSystemModel {
  public:
    KFSampleSystemModel(uint16_t n, uint16_t m, uint16_t p)
        : LinearSystemModel::LinearSystemModel(n, m, p) {
        setA(Eigen::MatrixXd::Identity(state_dims_, state_dims_));
        R_p_ = 1e-3 * Eigen::MatrixXd::Identity(state_dims_, state_dims_);
        P_ = Eigen::MatrixXd::Identity(state_dims_, state_dims_);
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override {
        setB(dt * Eigen::MatrixXd::Identity(x.size(), x.size()));
    }
};

class KFSampleMeasurementModel : public measurement_models::LinearMeasurementModel {
  public:
    KFSampleMeasurementModel(uint16_t n, uint16_t k)
        : LinearMeasurementModel::LinearMeasurementModel(n, k) {
        setC(Eigen::MatrixXd::Identity(meas_dims_, state_dims_));
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override {}
};

template <typename FilterT>
class KalmanFilterTestT
    : public FilterTest<FilterT, KFSampleSystemModel, KFSampleMeasurementModel> {
  public:
    void predictUsesAxPlusBuForStateUpdate() {
        double dt = 0.4;

        // Modify our A and B matrices
        Eigen::MatrixXd A(2, 2);
        A << 1, 2, 3, 4;
        Eigen::MatrixXd B = dt * Eigen::MatrixXd::Identity(2, 2);

        // Run it through the filter with the updated matrices
        this->system_model.setA(A);
        this->system_model.setB(B);
        this->filter->predict(this->vec_22, this->filter->getStateTime() + dt);

        // Our new state should be: x' = Ax + Bu
        Eigen::VectorXd x_target = A * this->x_i + B * this->vec_22;

        EXPECT_TRUE(this->filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose()
            << ", Actual: " << this->filter->getState().transpose();
    }

    void predictUsesAMatrixForCovarianceUpdate() {
        double dt = 0.4;

        // Modify our A matrix
        Eigen::MatrixXd A(2, 2);
        A << 1, 2, 3, 4;

        // Run it through the filter, with zero process noise to isolate the Jacobian
        this->system_model.setA(A);
        this->system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));
        this->filter->predict(this->vec_22, this->filter->getStateTime() + dt);

        // Compute the target covariance with the Kalman update of Sigma = A * Sigma * A'
        Eigen::MatrixXd cov_target = A * this->cov_i * A.transpose();

        EXPECT_TRUE(this->filter->getCovariance().isApprox(cov_target, 1e-6))
            << "Target:" << std::endl
            << cov_target << std::endl
            << "Actual:" << std::endl
            << this->filter->getCovariance();
    }
};

/////////////////////////////////
// EKF test utilities

template <typename FilterT>
class EKFTestT : public FilterTest<FilterT, SampleSystemModel<1>, SampleMeasurementModel<1>> {
  public:
    void predictUses_g_FunctionForStateUpdate() {
        double dt = 0.4;

        // Run it through the filter
        this->filter->predict(this->vec_22, this->filter->getStateTime() + dt);

        // Run it through the model
        SampleSystemModel<1> eval_model(2, 2, 2);
        eval_model.update(this->x_i, this->vec_22, dt);
        Eigen::VectorXd x_target = eval_model.g();

        EXPECT_TRUE(this->filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose()
            << ", Actual: " << this->filter->getState().transpose();
    }

    void predictUses_G_FunctionForCovarianceUpdate() {
        double dt = 0.4;

        // Run it through the filter, with zero process noise to isolate the Jacobian
        this->system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));
        this->filter->predict(this->vec_22, this->filter->getStateTime() + dt);

        // Compute the target covariance with the EKF update of Sigma = G * Sigma * G'
        SampleSystemModel<1> eval_model(2, 2, 2);
        eval_model.update(this->x_i, this->vec_22, dt);
        Eigen::MatrixXd cov_target = eval_model.G() * this->cov_i * eval_model.G().transpose();

        EXPECT_TRUE(this->filter->getCovariance().isApprox(cov_target, 1e-6))
            << "Target:" << std::endl
            << cov_target << std::endl
            << "Actual:" << std::endl
            << this->filter->getCovariance();
    }
};

/////////////////////////////////
// UKF test utilities

template <typename FilterT>
class UKFTestT : public FilterTest<FilterT, SampleSystemModel<1>, SampleMeasurementModel<1>> {
  public:
    void predictUses_g_FunctionForStateUpdate() {
        double dt = 0.4;

        // Run it through the filter
        this->filter->predict(this->vec_22, this->filter->getStateTime() + dt);

        // Run it through the model
        SampleSystemModel<1> eval_model(2, 2, 2);
        eval_model.update(this->x_i, this->vec_22, dt);
        Eigen::VectorXd x_target = eval_model.g();

        EXPECT_TRUE(this->filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose()
            << ", Actual: " << this->filter->getState().transpose();
    }

    void predictUsesSystemModelProcessNoise() {
        double dt = 0.4;

        // Create two filters with two different system models that have different process noise
        // levels
        Eigen::MatrixXd sigma_1 = 1e-2 * Eigen::MatrixXd::Identity(2, 2);
        SampleSystemModel<1> model_1(2, 2, 2);
        model_1.setProcessCovariance(sigma_1);
        FilterT filter_1(&model_1, this->x_i, this->cov_i, this->t_i);
        filter_1.predict(this->vec_22, this->t_i + dt);

        Eigen::MatrixXd sigma_2 = 1e-4 * Eigen::MatrixXd::Identity(2, 2);
        SampleSystemModel<1> model_2(2, 2, 2);
        model_2.setProcessCovariance(sigma_2);
        FilterT filter_2(&model_2, this->x_i, this->cov_i, this->t_i);
        filter_2.predict(this->vec_22, this->t_i + dt);

        // The covariance between the two filters should only differ by the difference between the
        // two process noise levels since the process noise is added on top of the system update
        Eigen::MatrixXd cov_diff = filter_2.getCovariance() - filter_1.getCovariance();
        Eigen::MatrixXd target_diff = sigma_2 - sigma_1;

        EXPECT_TRUE(cov_diff.isApprox(target_diff, 1e-6)) << "Target:" << std::endl
                                                          << target_diff << std::endl
                                                          << "Actual:" << std::endl
                                                          << cov_diff;
    }

    // The four tests below all attempt to exploit how the UKF handles non linearities. Although we
    // can't verify the exact state and covariance values after being passed through the non linear
    // functions without re implementing the UKF equations again here, we can make sure the outputs
    // move around in the right direction with some known non linearities.
    //
    // We'll use non linear variants of our sample models. Since our mock system model applies
    // an exponent to the current state we can use that to make the the updates for different
    // sigma points quite different from each other.

    void predictionMeanShiftsWithNonLinearity() {
        // An exponent of 3 in the system model should produce the mean when the state is centered
        // about zero, but when the state is shifted off center the updated state should shift a lot
        // in that direction.
        double dt = 0.1;
        SampleSystemModel<3> nl_sys_model(2, 2, 2);
        FilterT ukf(&nl_sys_model);

        ukf.initialize(this->vec_00, this->cov_i, this->t_i);
        ukf.predict(this->vec_22, this->t_i + dt);

        // TODO: FIX THIS!!

        // Target is an identity plus the control scaled by time
        Eigen::VectorXd x_target = dt * (this->vec_11 + this->vec_22);
        EXPECT_TRUE(ukf.getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose() << ", Actual: " << ukf.getState().transpose();

        // Reset, but with the initial state shifted slightly off the origin in the positive
        // direction
        ukf.initialize(this->vec_11, this->cov_i, this->t_i);
        ukf.predict(this->vec_22, this->t_i + dt);

        EXPECT_GT(ukf.getState()(0), x_target(0));
        EXPECT_GT(ukf.getState()(1), x_target(1));

        // Reset, but with the initial state shifted slightly off the origin in the negative
        // direction
        ukf.initialize(this->vec_n11, this->cov_i, this->t_i);
        ukf.predict(this->vec_22, this->t_i + dt);

        EXPECT_LT(ukf.getState()(0), x_target(0));
        EXPECT_LT(ukf.getState()(1), x_target(1));
    }

    void predictionCovarianceChangesWithNonLinearity() {
        // As the exponent in the system model becomes larger the non linearity should be more
        // exaggerated so the sigma points post transform should be much further away from each
        // other so the covariance should grow.
        double dt = 0.1;

        SampleSystemModel<1> model_1(2, 2, 2);
        FilterT ukf_1(&model_1, this->vec_22, this->cov_i, this->t_i);
        ukf_1.predict(this->vec_22, this->t_i + dt);

        SampleSystemModel<2> model_2(2, 2, 2);
        FilterT ukf_2(&model_2, this->vec_22, this->cov_i, this->t_i);
        ukf_2.predict(this->vec_22, this->t_i + dt);

        SampleSystemModel<3> model_3(2, 2, 2);
        FilterT ukf_3(&model_3, this->vec_22, this->cov_i, this->t_i);
        ukf_3.predict(this->vec_22, this->t_i + dt);

        EXPECT_GT(ukf_2.getCovariance().norm(), ukf_1.getCovariance().norm())
            << "C2=" << std::endl
            << ukf_2.getCovariance() << std::endl
            << "C1=" << std::endl
            << ukf_1.getCovariance() << std::endl;
        EXPECT_GT(ukf_3.getCovariance().norm(), ukf_2.getCovariance().norm())
            << "C3=" << std::endl
            << ukf_3.getCovariance() << std::endl
            << "C2=" << std::endl
            << ukf_2.getCovariance() << std::endl;
    }

    void correctionMeanShiftsWithNonLinearity() {
        // An exponent of 3 in the system model should produce the mean when the state is centered
        // about zero, but when the state is shifted off center the updated state should shift in
        // the opposite direction (because the predicted measurement will be greater in magnitude
        // than the actual measurement)
        SampleMeasurementModel<3> nl_meas_model(2, 2);
        this->filter->initialize(this->vec_00, this->cov_i, this->t_i);

        this->filter->correct(this->vec_00, this->cov_i, this->t_i, &nl_meas_model);

        // Target is the measurement supplied
        Eigen::VectorXd x_target = this->vec_00;
        EXPECT_TRUE(this->filter->getState().isApprox(x_target, 1e-6))
            << "Target: " << x_target.transpose()
            << ", Actual: " << this->filter->getState().transpose();

        // Reset, but with the initial state shifted slightly off the origin in the positive
        // direction
        this->filter->initialize(this->vec_22, this->cov_i, this->t_i);
        this->filter->correct(this->vec_22, this->cov_i, this->t_i, &nl_meas_model);

        EXPECT_LT(this->filter->getState()(0), this->vec_22(0));
        EXPECT_LT(this->filter->getState()(1), this->vec_22(1));

        // Reset, but with the initial state shifted slightly off the origin in the negative
        // direction
        this->filter->initialize(this->vec_n22, this->cov_i, this->t_i);
        this->filter->correct(this->vec_n22, this->cov_i, this->t_i, &nl_meas_model);

        EXPECT_GT(this->filter->getState()(0), this->vec_n22(0));
        EXPECT_GT(this->filter->getState()(1), this->vec_n22(1));
    }

    void correctionCovarianceChangesWithNonLinearity() {
        // As the exponent in the system model becomes larger the non linearity should be more
        // exaggerated so the sigma points post transform should be much further away from each
        // other and the covariance will decrease because these measurements are more informative.
        Eigen::VectorXd x = this->vec_22;

        SampleMeasurementModel<1> nl_meas_model_1(2, 2);
        this->filter->initialize(x, this->cov_i, this->t_i);
        this->filter->correct(x, this->cov_i, this->t_i, &nl_meas_model_1);
        Eigen::MatrixXd cov_1 = this->filter->getCovariance();

        SampleMeasurementModel<2> nl_meas_model_2(2, 2);
        this->filter->initialize(x, this->cov_i, this->t_i);
        this->filter->correct(x, this->cov_i, this->t_i, &nl_meas_model_2);
        Eigen::MatrixXd cov_2 = this->filter->getCovariance();

        SampleMeasurementModel<4> nl_meas_model_3(2, 2);
        this->filter->initialize(x, this->cov_i, this->t_i);
        this->filter->correct(x, this->cov_i, this->t_i, &nl_meas_model_3);
        Eigen::MatrixXd cov_3 = this->filter->getCovariance();

        EXPECT_LT(cov_2.norm(), cov_1.norm()) << "cov_2=" << std::endl
                                              << cov_2 << std::endl
                                              << "cov_1=" << std::endl
                                              << cov_1 << std::endl;
        EXPECT_LT(cov_3.norm(), cov_2.norm()) << "cov_3=" << std::endl
                                              << cov_3 << std::endl
                                              << "cov_2=" << std::endl
                                              << cov_2 << std::endl;
    }
};
