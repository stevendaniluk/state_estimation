#include <gtest/gtest.h>
#include <state_estimation/utilities/angle_utilities.h>

using namespace state_estimation;

TEST(AngleUtilitiesAngleDifference, NearZero) {
    double theta1 = 0.1 * M_PI;
    double theta2 = -0.1 * M_PI;

    EXPECT_FLOAT_EQ(0.2 * M_PI, angleDifference(theta1, theta2));
    EXPECT_FLOAT_EQ(-0.2 * M_PI, angleDifference(theta2, theta1));
}

TEST(AngleUtilitiesAngleDifference, NearCrossover) {
    double theta1 = 0.9 * M_PI;
    double theta2 = -0.9 * M_PI;

    EXPECT_FLOAT_EQ(-0.2 * M_PI, angleDifference(theta1, theta2));
    EXPECT_FLOAT_EQ(0.2 * M_PI, angleDifference(theta2, theta1));
}

TEST(AngleUtilitiesWeightedAngleSum, EqualAnglesProducesSameAngle) {
    // Make all the angles the same, but the weights different
    Eigen::VectorXd angles = Eigen::VectorXd::Constant(5, 0.123);
    Eigen::VectorXd w(5);
    w << 0.1, 0.2, 0.3, 0.4, 0.5;

    EXPECT_NEAR(0.123, weightedAngleSum(w, angles), 1e-5);
}

TEST(AngleUtilitiesWeightedAngleSum, EqualWeightsProducesMean) {
    double target = 0.345;
    double offset = 0.123;

    Eigen::VectorXd angles(4);
    angles << (target - 2 * offset), (target - offset), (target + offset), (target + 2 * offset);

    Eigen::VectorXd w = Eigen::VectorXd::Ones(4);

    EXPECT_NEAR(target, weightedAngleSum(w, angles), 1e-5);
}

TEST(AngleUtilitiesWeightedAngleSum, UnequalWeights) {
    // Setup two angles and weights so that the result is a linear interpolation between them
    Eigen::VectorXd angles(2);
    angles << 0.1, 0.2;

    Eigen::VectorXd w(2);
    w << 0.4, 0.6;

    EXPECT_NEAR(0.16, weightedAngleSum(w, angles), 1e-5);
}
