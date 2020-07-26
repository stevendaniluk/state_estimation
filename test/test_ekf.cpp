#include <gtest/gtest.h>
#include <state_estimation/filters/ekf.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the EKF class is only responsible for implementing the EKF update equations, only the
// correctness of those equations are tested

class EKFTest : public EKFTestT<EKF> {};

TEST_F(EKFTest, PredictUses_g_FunctionForStateUpdate) {
    predictUses_g_FunctionForStateUpdate();
}

TEST_F(EKFTest, PredictUses_G_FunctionForCovarianceUpdate) {
    predictUses_G_FunctionForCovarianceUpdate();
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
