#include <gtest/gtest.h>
#include <state_estimation/filters/kalman_filter.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the KalmanFilter class is only responsible for implementing the KF update equations, only
// the correctness of those equations are tested

class KalmanFilterTest : public KalmanFilterTestT<KalmanFilter> {};

TEST_F(KalmanFilterTest, PredictUsesAxPlusBuForStateUpdate) {
    predictUsesAxPlusBuForStateUpdate();
}

TEST_F(KalmanFilterTest, PredictUsesAMatrixForCovarianceUpdate) {
    predictUsesAMatrixForCovarianceUpdate();
}

TEST_F(KalmanFilterTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(KalmanFilterTest, ZeroMeasurementCovarianceProducesZeroEstimateCovariance) {
    zeroMeasurementCovarianceProducesZeroEstimateCovariance();
}

TEST_F(KalmanFilterTest, ZeroMeasurementCovarianceSetsStateToPredictedMeasurement) {
    zeroMeasurementCovarianceSetsStateToPredictedMeasurement();
}

TEST_F(KalmanFilterTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(KalmanFilterTest, CorrectWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement) {
    correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement();
}

TEST_F(KalmanFilterTest, CorrectWithEqualCovarianceHalvesTheCovarience) {
    correctWithEqualCovarianceHalvesTheCovarience();
}
