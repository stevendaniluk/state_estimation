#include <gtest/gtest.h>
#include <state_estimation/filters/ekf.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the EKF class is only responsible for implementing the EKF update equations, only the
// correctness of those equations are tested

// Pass in our filter, system and measurement types to the filter test fixture
class EKFTest : public FilterTest<EKF, SampleSystemModel<1>, SampleMeasurementModel<1>> {};

TEST_F(EKFTest, PredictUses_g_FunctionForStateUpdate) {
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

TEST_F(EKFTest, PredictUses_G_FunctionForCovarianceUpdate) {
    double dt = 0.4;

    // Run it through the filter, with zero process noise to isolate the Jacobian
    system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));
    filter->predict(vec_22, filter->getStateTime() + dt);

    // Compute the target covariance with the EKF update of Sigma = G * Sigma * G'
    SampleSystemModel<1> eval_model(2, 2, 2);
    eval_model.update(x_i, vec_22, dt);
    Eigen::MatrixXd cov_target = eval_model.G() * cov_i * eval_model.G().transpose();

    EXPECT_TRUE(filter->getCovariance().isApprox(cov_target, 1e-6)) << "Target:" << std::endl
                                                                    << cov_target << std::endl
                                                                    << "Actual:" << std::endl
                                                                    << filter->getCovariance();
}

TEST_F(EKFTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(EKFTest, ZeroMeasurementCovarianceProducesZeroEstimateCovariance) {
    zeroMeasurementCovarianceProducesZeroEstimateCovariance();
}

TEST_F(EKFTest, ZeroMeasurementCovarianceSetsStateToPredictedMeasurement) {
    zeroMeasurementCovarianceSetsStateToPredictedMeasurement();
}

TEST_F(EKFTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(EKFTest, CorrectWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement) {
    correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement();
}

TEST_F(EKFTest, CorrectWithEqualCovarianceHalvesTheCovarience) {
    correctWithEqualCovarianceHalvesTheCovarience();
}

TEST_F(EKFTest, PredictOnlyUpdatesActiveStates) {
    predictOnlyUpdatesActiveStates();
}

TEST_F(EKFTest, PredictOnlyUsesActiveControls) {
    predictOnlyUsesActiveControls();
}

TEST_F(EKFTest, CorrectOnlyUpdatesActiveStates) {
    correctOnlyUpdatesActiveStates();
}

TEST_F(EKFTest, CorrectOnlyUsesActiveMeasurements) {
    correctOnlyUsesActiveMeasurements();
}
