#include <gtest/gtest.h>
#include <state_estimation/filters/ukf.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the UKF class is only responsible for implementing the UKF update equations, only the
// correctness of those equations are tested

TEST_F(UKFTest, PredictUses_g_FunctionForStateUpdate) {
    predictUses_g_FunctionForStateUpdate();
}

TEST_F(UKFTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(UKFTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(UKFTest, PredictionMeanShiftsWithNonLinearity) {
    predictionMeanShiftsWithNonLinearity();
}

TEST_F(UKFTest, PredictionCovarianceChangesWithNonLinearity) {
    predictionCovarianceChangesWithNonLinearity();
}

TEST_F(UKFTest, CorrectionMeanShiftsWithNonLinearity) {
    correctionMeanShiftsWithNonLinearity();
}

TEST_F(UKFTest, CorrectionCovarianceChangesWithNonLinearity) {
    correctionCovarianceChangesWithNonLinearity();
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
