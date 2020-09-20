#include <gtest/gtest.h>
#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/system_models/six_d_rates.h>
#include "model_test_utilities.h"

using namespace state_estimation;
using namespace six_d_rates;

TEST(SixDRates, VelocityIntegration) {
    double dt = 0.2;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(state::DIMS);
    x(state::AX) = 3.14159;
    x(state::AY) = 2.71828;
    x(state::AZ) = 1.41421;
    Eigen::VectorXd u(0);

    system_models::SixDRates model;
    model.update(x, u, dt);

    Eigen::Vector3d v_target = dt * x.segment(state::AX, 3);
    EXPECT_FLOAT_EQ(v_target.x(), model.g()(state::VX));
    EXPECT_FLOAT_EQ(v_target.y(), model.g()(state::VY));
    EXPECT_FLOAT_EQ(v_target.z(), model.g()(state::VZ));
}

TEST(SixDOmniAccel, JacobianMatchesNumericalApproximation) {
    system_models::SixDRates model;
    double dt = 0.01;
    Eigen::VectorXd x(state::DIMS);
    x.segment(state::VX, 3) << 1.1, 2.2, 3.3;
    x.segment(state::AX, 3) << 4.4, 5.5, 6.6;
    x.segment(state::VPHI, 3) << 0.111, 0.222, 0.333;
    x.segment(state::GX, 3) << 0.123, 0.456, -0.9;
    x.segment(state::B_AX, 3) << 0.0011, 0.0022, 0.0033;
    x.segment(state::B_WX, 3) << -0.0044, 0.0055, 0.0066;

    Eigen::VectorXd u(0);

    jacobianMatchesNumericalApproximation(&model, x, u, dt);
}

TEST(SixDOmniAccel, InactiveStatesDoNotChange) {
    system_models::SixDRates model;
    double dt = 0.2;
    Eigen::VectorXd x = Eigen::VectorXd::Random(state::DIMS);
    Eigen::VectorXd u(0);

    inactiveStatesDoNotChange(&model, x, u, dt);
}
