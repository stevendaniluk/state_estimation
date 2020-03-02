#include <gtest/gtest.h>
#include <state_estimation/filters/ukf.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Pass in our filter, system and measurement types to the filter test fixture
class UKFTest : public FilterTest<UKF, SampleSystemModel<1>, SampleMeasurementModel<1>> {};

TEST_F(UKFTest, PredictUses_g_FunctionForStateUpdate) {
    double dt = 0.4;

    // Run it through the filter
    filter->predict(vec_22, filter->getStateTime() + dt);

    // Run it through the model
    SampleSystemModel<1> eval_model(2, 2, 2);
    eval_model.update(x_i, vec_22, dt);
    Eigen::VectorXd x_target = eval_model.g();

    EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(UKFTest, PredictUsesSystemModelProcessNoise) {
    double dt = 0.4;

    // Create two filters with two different system models that have different process noise
    // levels
    Eigen::MatrixXd sigma_1 = 1e-2 * Eigen::MatrixXd::Identity(2, 2);
    SampleSystemModel<1> model_1(2, 2, 2);
    model_1.setProcessCovariance(sigma_1);
    UKF filter_1(&model_1, x_i, cov_i, t_i);
    filter_1.predict(vec_22, t_i + dt);

    Eigen::MatrixXd sigma_2 = 1e-4 * Eigen::MatrixXd::Identity(2, 2);
    SampleSystemModel<1> model_2(2, 2, 2);
    model_2.setProcessCovariance(sigma_2);
    UKF filter_2(&model_2, x_i, cov_i, t_i);
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

TEST_F(UKFTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

// The four tests below all attempt to exploit how the UKF handles non linearities. Although we
// can't verify the exact state and covariance values after being passed through the non linear
// functions without re implementing the UKF equations again here, we can make sure the outputs
// move around in the right direction with some known non linearities.
//
// We'll use non linear variants of our sample models. Since our mock system model applies
// an exponent to the current state we can use that to make the the updates for different
// sigma points quite different from each other.

TEST_F(UKFTest, PredictionMeanShiftsWithNonLinearity) {
    // An exponent of 3 in the system model should produce the mean when the state is centered
    // about zero, but when the state is shifted off center the updated state should shift a lot in
    // that direction.
    double dt = 0.1;
    SampleSystemModel<3> nl_sys_model(2, 2, 2);
    UKF ukf(&nl_sys_model);

    ukf.initialize(vec_00, cov_i, t_i);
    ukf.predict(vec_22, t_i + dt);

    // TODO: FIX THIS!!

    // Target is an identity plus the control scaled by time
    Eigen::VectorXd x_target = dt * (vec_11 + vec_22);
    EXPECT_TRUE(ukf.getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << ukf.getState().transpose();

    // Reset, but with the initial state shifted slightly off the origin in the positive direction
    ukf.initialize(vec_11, cov_i, t_i);
    ukf.predict(vec_22, t_i + dt);

    EXPECT_GT(ukf.getState()(0), x_target(0));
    EXPECT_GT(ukf.getState()(1), x_target(1));

    // Reset, but with the initial state shifted slightly off the origin in the negative direction
    ukf.initialize(vec_n11, cov_i, t_i);
    ukf.predict(vec_22, t_i + dt);

    EXPECT_LT(ukf.getState()(0), x_target(0));
    EXPECT_LT(ukf.getState()(1), x_target(1));
}

TEST_F(UKFTest, PredictionCovarianceChangesWithNonLinearity) {
    // As the exponent in the system model becomes larger the non linearity should be more
    // exaggerated so the sigma points post transform should be much further away from each other
    // so the covariance should grow.
    double dt = 0.1;

    SampleSystemModel<1> model_1(2, 2, 2);
    UKF ukf_1(&model_1, vec_22, cov_i, t_i);
    ukf_1.predict(vec_22, t_i + dt);

    SampleSystemModel<2> model_2(2, 2, 2);
    UKF ukf_2(&model_2, vec_22, cov_i, t_i);
    ukf_2.predict(vec_22, t_i + dt);

    SampleSystemModel<3> model_3(2, 2, 2);
    UKF ukf_3(&model_3, vec_22, cov_i, t_i);
    ukf_3.predict(vec_22, t_i + dt);

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

TEST_F(UKFTest, CorrectionMeanShiftsWithNonLinearity) {
    // An exponent of 3 in the system model should produce the mean when the state is centered
    // about zero, but when the state is shifted off center the updated state should shift in the
    // opposite direction (because the predicted measurement will be greater in magnitude than
    // the actual measurement)
    SampleMeasurementModel<3> nl_meas_model(2, 2);
    filter->initialize(vec_00, cov_i, t_i);

    filter->correct(vec_00, cov_i, t_i, &nl_meas_model);

    // Target is the measurement supplied
    Eigen::VectorXd x_target = vec_00;
    EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();

    // Reset, but with the initial state shifted slightly off the origin in the positive direction
    filter->initialize(vec_22, cov_i, t_i);
    filter->correct(vec_22, cov_i, t_i, &nl_meas_model);

    EXPECT_LT(filter->getState()(0), vec_22(0));
    EXPECT_LT(filter->getState()(1), vec_22(1));

    // Reset, but with the initial state shifted slightly off the origin in the negative direction
    filter->initialize(vec_n22, cov_i, t_i);
    filter->correct(vec_n22, cov_i, t_i, &nl_meas_model);

    EXPECT_GT(filter->getState()(0), vec_n22(0));
    EXPECT_GT(filter->getState()(1), vec_n22(1));
}

TEST_F(UKFTest, CorrectionCovarianceChangesWithNonLinearity) {
    // As the exponent in the system model becomes larger the non linearity should be more
    // exaggerated so the sigma points post transform should be much further away from each other
    // and the covariance will decrease because these measurements are more informative.
    Eigen::VectorXd x = vec_22;

    SampleMeasurementModel<1> nl_meas_model_1(2, 2);
    filter->initialize(x, cov_i, t_i);
    filter->correct(x, cov_i, t_i, &nl_meas_model_1);
    Eigen::MatrixXd cov_1 = filter->getCovariance();

    SampleMeasurementModel<2> nl_meas_model_2(2, 2);
    filter->initialize(x, cov_i, t_i);
    filter->correct(x, cov_i, t_i, &nl_meas_model_2);
    Eigen::MatrixXd cov_2 = filter->getCovariance();

    SampleMeasurementModel<4> nl_meas_model_3(2, 2);
    filter->initialize(x, cov_i, t_i);
    filter->correct(x, cov_i, t_i, &nl_meas_model_3);
    Eigen::MatrixXd cov_3 = filter->getCovariance();

    EXPECT_LT(cov_2.norm(), cov_1.norm()) << "cov_2=" << std::endl
                                          << cov_2 << std::endl
                                          << "cov_1=" << std::endl
                                          << cov_1 << std::endl;
    EXPECT_LT(cov_3.norm(), cov_2.norm()) << "cov_3=" << std::endl
                                          << cov_3 << std::endl
                                          << "cov_2=" << std::endl
                                          << cov_2 << std::endl;
}

TEST_F(UKFTest, PredictOnlyUpdatesActiveStates) {
    predictOnlyUpdatesActiveStates();
}

TEST_F(UKFTest, PredictOnlyUsesActiveControls) {
    predictOnlyUsesActiveControls();
}

TEST_F(UKFTest, CorrectOnlyUpdatesActiveStates) {
    correctOnlyUpdatesActiveStates();
}

TEST_F(UKFTest, CorrectOnlyUsesActiveMeasurements) {
    correctOnlyUsesActiveMeasurements();
}
