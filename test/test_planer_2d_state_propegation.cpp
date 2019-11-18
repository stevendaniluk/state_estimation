#include <gtest/gtest.h>
#include <state_estimation/system_models/planer_2d_state_propegation.h>
#include "model_test_utilities.h"

using namespace state_estimation;
using namespace planer_2d;

TEST(Planer2DStatePropegation, FailsWithNonEmptyControl) {
    double dt = 0.2;
    Eigen::VectorXd u(1);
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.111, 0.222;

    system_models::Planer2DStatePropegation model;
    EXPECT_DEATH(model.update(x, u, dt), "");
}

TEST(Planer2DStatePropegation, LinearAccelerationIsDirectlyCopied) {
    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.111, 0.222;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    EXPECT_FLOAT_EQ(x(state::AX), model.g()(state::AX));
    EXPECT_FLOAT_EQ(x(state::AY), model.g()(state::AY));
}

TEST(Planer2DStatePropegation, AngularVelocityIsDirectlyCopied) {
    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.111, 0.222;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    EXPECT_FLOAT_EQ(x(state::VPSI), model.g()(state::VPSI));
}

TEST(Planer2DStatePropegation, HeadingUpdatedWithRate) {
    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.111, 0.222;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    double target_psi = x(state::PSI) + dt * x(state::VPSI);
    Eigen::VectorXd x_pred = model.g();
    EXPECT_FLOAT_EQ(x_pred(state::PSI), target_psi);
}

TEST(Planer2DStatePropegation, HeadingWithinIntervalPlusMinusPi) {
    // Setup an initial state that will cross the pi boundary
    double dt = 2.0;
    Eigen::VectorXd x(state::DIMS);
    x << 0, 0, 0, 0, 0, 0, M_PI - 0.1, 0.1;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    double target_psi = -M_PI + 0.1;

    Eigen::VectorXd x_pred = model.g();
    EXPECT_FLOAT_EQ(x_pred(state::PSI), target_psi);
}

TEST(Planer2DStatePropegation, PositionUpdatedWithConstantAccelerationModel) {
    // Create a state with zero heading and yaw rate, so we don't have to worry about state
    // variables being in different frames or orientation.
    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0, 0;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    Eigen::VectorXd pos = x.segment(state::X, 2);
    Eigen::VectorXd vel = x.segment(state::VX, 2);
    Eigen::VectorXd acc = x.segment(state::AX, 2);

    Eigen::VectorXd target_pos = pos + dt * vel + 0.5 * dt * dt * acc;
    Eigen::VectorXd actual_pos = model.g().segment(state::X, 2);

    EXPECT_TRUE(actual_pos.isApprox(target_pos, 1e-3))
        << "Target: " << target_pos.transpose() << ", Actual: " << actual_pos.transpose();
}

TEST(Planer2DStatePropegation, VelocityInInertialFrame) {
    // The velocity should be defined relative to the current orientation. Check this by manually
    // rotating our velocity vector and adding that motion to the state. Should get the same result.
    // We'll want to zero the acceleration to isolate the velocity update.
    double psi = 0.123;
    Eigen::Rotation2D<double> rot(psi);

    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 0, 0, psi, 0;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    Eigen::VectorXd target_pos = x.segment(state::X, 2) + dt * (rot * x.segment(state::VX, 2));
    Eigen::VectorXd actual_pos = model.g().segment(state::X, 2);

    EXPECT_TRUE(actual_pos.isApprox(target_pos, 1e-3))
        << "Target: " << target_pos.transpose() << ", Actual: " << actual_pos.transpose();
}

TEST(Planer2DStatePropegation, AccelerationInInertialFrame) {
    // The acceleration should be defined relative to the current orientation. Check this by
    // manually rotating our acceleration vector and adding that motion to the state. Should get
    // the same result. We'll want to zero the velocity to isolate the acceleration update.
    double psi = 0.123;
    Eigen::Rotation2D<double> rot(psi);

    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 0, 0, 3.3, 4.4, psi, 0;

    system_models::Planer2DStatePropegation model;
    model.updateNoControl(x, dt);

    Eigen::VectorXd target_pos =
        x.segment(state::X, 2) + 0.5 * dt * dt * (rot * x.segment(state::AX, 2));
    Eigen::VectorXd actual_pos = model.g().segment(state::X, 2);

    EXPECT_TRUE(actual_pos.isApprox(target_pos, 1e-3))
        << "Target: " << target_pos.transpose() << ", Actual: " << actual_pos.transpose();
}

TEST(Planer2DStatePropegation, JacobianMatchesNumericalApproximation) {
    system_models::Planer2DStatePropegation model(true);
    double dt = 0.2;
    Eigen::VectorXd x(state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.111, 0.222;

    jacobianMatchesNumericalApproximation(&model, x, dt);
}
