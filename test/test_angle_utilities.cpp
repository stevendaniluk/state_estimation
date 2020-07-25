#include <gtest/gtest.h>
#include <state_estimation/utilities/angle_utilities.h>

using namespace state_estimation;

void quaternionEqual(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2,
                     double epsilon = 1e-4) {
    bool w = fabs(q1.w() - q2.w()) < epsilon || fabs(q1.w() + q2.w()) < epsilon;
    bool x = fabs(q1.x() - q2.x()) < epsilon || fabs(q1.x() + q2.x()) < epsilon;
    bool y = fabs(q1.y() - q2.y()) < epsilon || fabs(q1.y() + q2.y()) < epsilon;
    bool z = fabs(q1.z() - q2.z()) < epsilon || fabs(q1.z() + q2.z()) < epsilon;

    EXPECT_TRUE(w && x && y && z) << "q1=[" << q1.w() << " " << q1.vec().transpose() << "], q2=["
                                  << q2.w() << " " << q2.vec().transpose() << "]";
}

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

TEST(AngleUtilitiesQuaternionAverage, FullWeighting) {
    // Create two sample quaternions
    Eigen::Quaterniond q1(0.1584234, 0.7116008, 0.4059919, 0.5510871);
    Eigen::Quaterniond q2(0.5200544, -0.2415334, -0.6404592, 0.5108982);
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> q_set = {q1, q2};

    Eigen::VectorXd w(2);

    w << 1, 0;
    Eigen::Quaterniond q_avg_1 = weightedQuaternion(w, q_set);
    quaternionEqual(q1, q_avg_1);

    w << 0, 1;
    Eigen::Quaterniond q_avg_2 = weightedQuaternion(w, q_set);
    quaternionEqual(q2, q_avg_2);
}

TEST(AngleUtilitiesQuaternionAverage, KnownAverage) {
    // Make a sample quaternion and apply the rotation multiple times, then weight them to
    // arrive at the successive rotations
    Eigen::Quaterniond q1(0.9987503, 0.0133575, 0.026715, 0.0400725);
    Eigen::Quaterniond q2 = q1 * q1;
    Eigen::Quaterniond q3 = q2 * q1;
    Eigen::Quaterniond q4 = q3 * q1;
    Eigen::Quaterniond q5 = q4 * q1;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> q_set = {
        q1, q2, q3, q4, q5};

    Eigen::VectorXd w(5);

    // Mean of first three should be the second orientation
    w << 0.3333, 0.3333, 0.3333, 0.0, 0.0;
    Eigen::Quaterniond q_avg_1 = weightedQuaternion(w, q_set);
    quaternionEqual(q2, q_avg_1);

    // Mean of all 5 should me the third orientation
    w << 0.2, 0.2, 0.2, 0.2, 0.2;
    Eigen::Quaterniond q_avg_2 = weightedQuaternion(w, q_set);
    quaternionEqual(q3, q_avg_2);
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
