#include <gtest/gtest.h>
#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/measurement_models/six_d_rates_imu.h>
#include "model_test_utilities.h"

using namespace state_estimation;
using namespace six_d_rates;

TEST(SixDRatesImuMeasurementModel, BiasesSubtracted) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::AX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::B_AX, 3) << 0.00123, 0.00456, 0.00789;

    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::B_WX, 3) << 0.00321, 0.00654, 0.00987;

    measurement_models::SixDRatesImu model;
    model.update(x, dt);
    Eigen::VectorXd z_pred = model.h();

    EXPECT_FLOAT_EQ(x(state::AX) - x(state::B_AX), z_pred(meas::imu::AX));
    EXPECT_FLOAT_EQ(x(state::AY) - x(state::B_AY), z_pred(meas::imu::AY));
    EXPECT_FLOAT_EQ(x(state::AZ) - x(state::B_AZ), z_pred(meas::imu::AZ));
    EXPECT_FLOAT_EQ(x(state::VPHI) - x(state::B_WX), z_pred(meas::imu::VPHI));
    EXPECT_FLOAT_EQ(x(state::VTHETA) - x(state::B_WY), z_pred(meas::imu::VTHETA));
    EXPECT_FLOAT_EQ(x(state::VPSI) - x(state::B_WZ), z_pred(meas::imu::VPSI));
}

TEST(SixDRatesImuMeasurementModel, GravityAdded) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::AX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::GX, 3) << 0.00123, 0.00456, 0.00789;

    measurement_models::SixDRatesImu model;
    model.update(x, dt);
    Eigen::VectorXd z_pred = model.h();

    EXPECT_FLOAT_EQ(x(state::AX) + x(state::GX), z_pred(meas::imu::AX));
    EXPECT_FLOAT_EQ(x(state::AY) + x(state::GY), z_pred(meas::imu::AY));
    EXPECT_FLOAT_EQ(x(state::AZ) + x(state::GZ), z_pred(meas::imu::AZ));
}

TEST(SixDRatesImuMeasurementModel, PredictedMeasurementAccountsForRotation) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::AX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;

    measurement_models::SixDRatesImu model;

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.123, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.456, Eigen::Vector3d::UnitX()));
    model.setTf(tf);

    model.update(x, dt);
    Eigen::VectorXd z_pred = model.h();

    Eigen::Vector3d a_target = tf.linear() * x.segment(state::AX, 3);
    Eigen::Vector3d w_target = tf.linear() * x.segment(state::VPHI, 3);

    EXPECT_FLOAT_EQ(a_target.x(), z_pred(meas::imu::AX));
    EXPECT_FLOAT_EQ(a_target.y(), z_pred(meas::imu::AY));
    EXPECT_FLOAT_EQ(a_target.z(), z_pred(meas::imu::AZ));
    EXPECT_FLOAT_EQ(w_target.x(), z_pred(meas::imu::VPHI));
    EXPECT_FLOAT_EQ(w_target.y(), z_pred(meas::imu::VTHETA));
    EXPECT_FLOAT_EQ(w_target.z(), z_pred(meas::imu::VPSI));
}

TEST(SixDRatesImuMeasurementModel, JacobianMatchesNumericalApproximation) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::AX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::B_AX, 3) << 0.00123, 0.00456, 0.00789;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::B_WX, 3) << 0.00321, 0.00654, 0.00987;
    x.segment(state::GX, 3) << 0.10, 0.20, -9.70;

    measurement_models::SixDRatesImu model;
    jacobianMatchesNumericalApproximation(&model, x, dt);
}

TEST(SixDRatesImuMeasurementModel, JacobianMatchesNumericalApproximationWithTransform) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::AX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::B_AX, 3) << 0.00123, 0.00456, 0.00789;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::B_WX, 3) << 0.00321, 0.00654, 0.00987;
    x.segment(state::GX, 3) << 0.10, 0.20, -9.70;

    measurement_models::SixDRatesImu model;

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.123, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.456, Eigen::Vector3d::UnitX()));
    model.setTf(tf);

    jacobianMatchesNumericalApproximation(&model, x, dt);
}
