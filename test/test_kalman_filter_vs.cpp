#include <gtest/gtest.h>
#include <state_estimation/filters/kalman_filter_vs.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the KalmanFilter class is only responsible for implementing the KF update equations, only
// the correctness of those equations are tested

class KalmanFilterVSTest : public KalmanFilterTestT<KalmanFilterVS> {};

TEST_F(KalmanFilterVSTest, PredictUsesAxPlusBuForStateUpdate) {
    predictUsesAxPlusBuForStateUpdate();
}

TEST_F(KalmanFilterVSTest, PredictUsesAMatrixForCovarianceUpdate) {
    predictUsesAMatrixForCovarianceUpdate();
}

TEST_F(KalmanFilterVSTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(KalmanFilterVSTest, ZeroMeasurementCovarianceProducesZeroEstimateCovariance) {
    zeroMeasurementCovarianceProducesZeroEstimateCovariance();
}

TEST_F(KalmanFilterVSTest, ZeroMeasurementCovarianceSetsStateToPredictedMeasurement) {
    zeroMeasurementCovarianceSetsStateToPredictedMeasurement();
}

TEST_F(KalmanFilterVSTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(KalmanFilterVSTest, CorrectWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement) {
    correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement();
}

TEST_F(KalmanFilterVSTest, CorrectWithEqualCovarianceHalvesTheCovarience) {
    correctWithEqualCovarianceHalvesTheCovarience();
}

TEST_F(KalmanFilterVSTest, PredictOnlyUpdatesActiveStates) {
    predictOnlyUpdatesActiveStates();
}

TEST_F(KalmanFilterVSTest, PredictOnlyUsesActiveControls) {
    predictOnlyUsesActiveControls();
}

TEST_F(KalmanFilterVSTest, CorrectOnlyUpdatesActiveStates) {
    correctOnlyUpdatesActiveStates();
}

TEST_F(KalmanFilterVSTest, CorrectOnlyUsesActiveMeasurements) {
    correctOnlyUsesActiveMeasurements();
}
