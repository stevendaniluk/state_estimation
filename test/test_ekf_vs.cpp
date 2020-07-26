#include <gtest/gtest.h>
#include <state_estimation/filters/ekf_vs.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the EKF class is only responsible for implementing the EKF update equations, only the
// correctness of those equations are tested

class EKFVSTest : public EKFTestT<EKFVS> {};

TEST_F(EKFVSTest, PredictUses_g_FunctionForStateUpdate) {
    predictUses_g_FunctionForStateUpdate();
}

TEST_F(EKFVSTest, PredictUses_G_FunctionForCovarianceUpdate) {
    predictUses_G_FunctionForCovarianceUpdate();
}

TEST_F(EKFVSTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(EKFVSTest, ZeroMeasurementCovarianceProducesZeroEstimateCovariance) {
    zeroMeasurementCovarianceProducesZeroEstimateCovariance();
}

TEST_F(EKFVSTest, ZeroMeasurementCovarianceSetsStateToPredictedMeasurement) {
    zeroMeasurementCovarianceSetsStateToPredictedMeasurement();
}

TEST_F(EKFVSTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(EKFVSTest, CorrectWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement) {
    correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement();
}

TEST_F(EKFVSTest, CorrectWithEqualCovarianceHalvesTheCovarience) {
    correctWithEqualCovarianceHalvesTheCovarience();
}

TEST_F(EKFVSTest, PredictOnlyUpdatesActiveStates) {
    predictOnlyUpdatesActiveStates();
}

TEST_F(EKFVSTest, PredictOnlyUsesActiveControls) {
    predictOnlyUsesActiveControls();
}

TEST_F(EKFVSTest, CorrectOnlyUpdatesActiveStates) {
    correctOnlyUpdatesActiveStates();
}

TEST_F(EKFVSTest, CorrectOnlyUsesActiveMeasurements) {
    correctOnlyUsesActiveMeasurements();
}
