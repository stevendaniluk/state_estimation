#include <gtest/gtest.h>
#include <state_estimation/filters/ukf_vs.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the UKF class is only responsible for implementing the UKF update equations, only the
// correctness of those equations are tested

class UKFVSTest : public UKFTestT<UKFVS> {};

TEST_F(UKFVSTest, PredictUses_g_FunctionForStateUpdate) {
    predictUses_g_FunctionForStateUpdate();
}

TEST_F(UKFVSTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(UKFVSTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(UKFVSTest, PredictionMeanShiftsWithNonLinearity) {
    predictionMeanShiftsWithNonLinearity();
}

TEST_F(UKFVSTest, PredictionCovarianceChangesWithNonLinearity) {
    predictionCovarianceChangesWithNonLinearity();
}

TEST_F(UKFVSTest, CorrectionMeanShiftsWithNonLinearity) {
    correctionMeanShiftsWithNonLinearity();
}

TEST_F(UKFVSTest, CorrectionCovarianceChangesWithNonLinearity) {
    correctionCovarianceChangesWithNonLinearity();
}

TEST_F(UKFVSTest, PredictOnlyUpdatesActiveStates) {
    predictOnlyUpdatesActiveStates();
}

TEST_F(UKFVSTest, PredictOnlyUsesActiveControls) {
    predictOnlyUsesActiveControls();
}

TEST_F(UKFVSTest, CorrectOnlyUpdatesActiveStates) {
    correctOnlyUpdatesActiveStates();
}

TEST_F(UKFVSTest, CorrectOnlyUsesActiveMeasurements) {
    correctOnlyUsesActiveMeasurements();
}
