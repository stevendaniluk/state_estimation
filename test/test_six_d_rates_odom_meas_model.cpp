#include <gtest/gtest.h>
#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/measurement_models/six_d_rates_odom.h>
#include "model_test_utilities.h"

using namespace state_estimation;
using namespace six_d_rates;

TEST(SixDRatesOdomMeasurementModel, PredictedMeasurementAccountsForRotation) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::VX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;

    measurement_models::SixDRatesOdom model;

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.123, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.456, Eigen::Vector3d::UnitX()));
    model.setTf(tf);

    model.update(x, dt);
    Eigen::VectorXd z_pred = model.h();

    // Linear ang angulare velocities should simply be rotated by the transformation
    Eigen::Vector3d v_target = tf.linear() * x.segment(state::VX, 3);
    Eigen::Vector3d w_target = tf.linear() * x.segment(state::VPHI, 3);

    EXPECT_FLOAT_EQ(v_target.x(), z_pred(meas::odom::VX));
    EXPECT_FLOAT_EQ(v_target.y(), z_pred(meas::odom::VY));
    EXPECT_FLOAT_EQ(v_target.z(), z_pred(meas::odom::VZ));
    EXPECT_FLOAT_EQ(w_target.z(), z_pred(meas::odom::VPSI));
}

TEST(SixDRatesOdomMeasurementModel, PredictedMeasurementAccountsForTranslation) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::VX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;

    measurement_models::SixDRatesOdom model;

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << 1.1, 2.2, 3.3;
    model.setTf(tf);

    model.update(x, dt);
    Eigen::VectorXd z_pred = model.h();

    Eigen::Vector3d v_state = x.segment(state::VX, 3);
    Eigen::Vector3d w_state = x.segment(state::VPHI, 3);

    // Linear velocity will have a contribution from the angular velocity due to the offset
    Eigen::Vector3d v_target = v_state + w_state.cross(tf.translation());
    Eigen::Vector3d w_target = w_state;

    EXPECT_FLOAT_EQ(v_target.x(), z_pred(meas::odom::VX));
    EXPECT_FLOAT_EQ(v_target.y(), z_pred(meas::odom::VY));
    EXPECT_FLOAT_EQ(v_target.z(), z_pred(meas::odom::VZ));
    EXPECT_FLOAT_EQ(w_target.z(), z_pred(meas::odom::VPSI));
}

TEST(SixDRatesOdomMeasurementModel, JacobianMatchesNumericalApproximation) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::VX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::AX, 3) << 0.314159, 0.271828, 0.141421;
    x.segment(state::B_AX, 3) << 0.00123, 0.00456, 0.00789;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::B_WX, 3) << 0.00321, 0.00654, 0.00987;
    x.segment(state::GX, 3) << 0.10, 0.20, -9.70;

    measurement_models::SixDRatesOdom model;
    jacobianMatchesNumericalApproximation(&model, x, dt);
}

TEST(SixDRatesOdomMeasurementModel, JacobianMatchesNumericalApproximationWithTransform) {
    double dt = 0.1;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x.segment(state::VX, 3) << 3.14159, 2.71828, 1.41421;
    x.segment(state::AX, 3) << 0.314159, 0.271828, 0.141421;
    x.segment(state::B_AX, 3) << 0.00123, 0.00456, 0.00789;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::B_WX, 3) << 0.00321, 0.00654, 0.00987;
    x.segment(state::GX, 3) << 0.10, 0.20, -9.70;

    measurement_models::SixDRatesOdom model;

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << 1.1, 2.2, 3.3;
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.123, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.456, Eigen::Vector3d::UnitX()));
    model.setTf(tf);

    jacobianMatchesNumericalApproximation(&model, x, dt);
}
