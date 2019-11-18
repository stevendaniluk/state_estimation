#include <gtest/gtest.h>
#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/measurement_models/planer_2d_imu.h>
#include "model_test_utilities.h"

using namespace state_estimation;

TEST(Planer2DImuMeasurementModel, StationaryStateProducesZeroMeasurement) {
    measurement_models::Planer2DImu model(false, false);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(planer_2d::state::DIMS);
    model.update(x);
    Eigen::VectorXd z_pred = model.h();

    Eigen::VectorXd z_target = Eigen::VectorXd::Zero(meas::imu::DIMS);

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-6))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(Planer2DImuMeasurementModel, StationaryStateWithGravity) {
    measurement_models::Planer2DImu model(false, true);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(planer_2d::state::DIMS);
    model.update(x);
    Eigen::VectorXd z_pred = model.h();

    Eigen::VectorXd z_target = Eigen::VectorXd::Zero(meas::imu::DIMS);
    z_target(meas::imu::AZ) = 9.80665;

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-3))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(Planer2DImuMeasurementModel, JacobianMatchesNumericalApproximation) {
    measurement_models::Planer2DImu model(true, false);
    Eigen::VectorXd x(planer_2d::state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.1, 0.2;

    jacobianMatchesNumericalApproximation(&model, x);
}

TEST(Planer2DImuMeasurementModel, JacobianMatchesNumericalApproximationWithGravity) {
    measurement_models::Planer2DImu model(true, true);
    Eigen::VectorXd x(planer_2d::state::DIMS);
    x << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.1, 0.2;

    jacobianMatchesNumericalApproximation(&model, x);
}
