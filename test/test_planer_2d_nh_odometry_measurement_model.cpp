#include <gtest/gtest.h>
#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/measurement_models/planer_2d_nh_odometry.h>
#include "model_test_utilities.h"

using namespace state_estimation;
using namespace planer_2d;

TEST(Planer2DNhOdometryMeasurementModel, PredictedMeasurementDirectlyFromState) {
    measurement_models::Planer2DNhOdometry model;

    Eigen::VectorXd x(state::DIMS);
    x(state::VX) = 3.14159;
    x(state::VPSI) = 2.71828;

    model.update(x);
    Eigen::VectorXd z_pred = model.h();

    EXPECT_FLOAT_EQ(x(state::VX), z_pred(meas::nh_odom::VX));
    EXPECT_FLOAT_EQ(x(state::VPSI), z_pred(meas::nh_odom::VPSI));
}

TEST(Planer2DNhOdometryMeasurementModel, JacobianUnityOnlyForMeasurementVariables) {
    measurement_models::Planer2DNhOdometry model;

    Eigen::VectorXd x(state::DIMS);
    x(state::VX) = 3.14159;
    x(state::VPSI) = 2.71828;

    model.update(x);
    Eigen::MatrixXd H = model.H();

    Eigen::MatrixXd H_target = Eigen::MatrixXd::Zero(meas::nh_odom::DIMS, state::DIMS);
    H_target(meas::nh_odom::VX, state::VX) = 1;
    H_target(meas::nh_odom::VPSI, state::VPSI) = 1;

    EXPECT_TRUE(H.isApprox(H_target, 1e-9)) << "Target:" << std::endl
                                            << H_target << std::endl
                                            << "Actual:" << H;
}

TEST(Planer2DImuMeasurementModel, JacobianMatchesNumericalApproximation) {
    measurement_models::Planer2DNhOdometry model;

    Eigen::VectorXd x(state::DIMS);
    x(state::VX) = 3.14159;
    x(state::VPSI) = 2.71828;

    jacobianMatchesNumericalApproximation(&model, x);
}
