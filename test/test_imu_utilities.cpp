#include <gtest/gtest.h>
#include <state_estimation/utilities/imu_utilities.h>
#include <state_estimation/utilities/transformation_utilities.h>

using namespace state_estimation;

// accelJacobianMatchesNumericalApproximation
//
// Helper for checking that the Jacobian produced by accelMeasurementJacobian() matches a
// numerically computed version.
//
// This helper makes use of the predictAccelMeasurement() method in the imu utilities library.
// Ideally we could numerically test the Jacobian method in isolation without making use of other
// things in the library, but unfortunately there isn't an easy way to do that.
//
// @param include_gravity: When true the change in acceleration due to orientation will be computed
//                         from the effect of gravity
// @param epsilon: Amount to perturb the state variables by
// @param tolerance: Tolerance on the Frobenius norm between the numerically determine Jacobian and
//                   the output of the system model
void accelJacobianMatchesNumericalApproximation(bool include_gravity, double epsilon = 1e-6,
                                                double tolerance = 1e-3) {
    const double gravity = 9.80665;

    // Form a state of the form [AX, AY, AZ, PHI, THETA, PSI, VPHI, VTHETA, VPSI], and a
    // transformation from the state frame to the accelerometer frame
    Eigen::Vector3d a(0.1, 0.2, 0.3);
    Eigen::Vector3d rpy(0.11, 0.22, 0.33);
    Eigen::Vector3d w(0.111, 0.222, 0.333);

    Eigen::Matrix<double, 9, 1> x_ref;
    x_ref << a, rpy, w;

    // Make up a transformation between the frames
    Eigen::Isometry3d tf;
    tf.translation() << 0.5, 0.1, 0.3;
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));

    // Compute the jacobian for the reference state
    Eigen::Matrix<double, 3, 9> H_target =
        accelMeasurementJacobian(w, tf, include_gravity, rpy, gravity);

    // Compute the expected measurement for the reference state
    Eigen::Quaterniond orientation =
        (Eigen::Quaterniond)Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::Matrix<double, 3, 1> z_pred =
        predictAccelMeasurement(a, w, tf, include_gravity, orientation, gravity);

    // Go through each element of the reference state perturbing the values slightly and computing
    // the new predicted measurements to numerically compute the jacobian
    Eigen::Matrix<double, 3, 9> H_num;
    for (int i = 0; i < 9; ++i) {
        Eigen::Matrix<double, 9, 1> x = x_ref;
        x(i) += epsilon;

        const Eigen::Vector3d a_pert = x.segment<3>(0);
        const Eigen::Vector3d rpy_pert = x.segment<3>(3);
        const Eigen::Vector3d w_pert = x.segment<3>(6);

        Eigen::Quaterniond orientation_pert =
            (Eigen::Quaterniond)Eigen::AngleAxisd(rpy_pert.z(), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(rpy_pert.y(), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rpy_pert.x(), Eigen::Vector3d::UnitX());

        Eigen::Matrix<double, 3, 1> z_pert =
            predictAccelMeasurement(a_pert, w_pert, tf, include_gravity, orientation_pert, 9.80665);

        // Compute the partial derivative
        H_num.col(i) = (z_pert - z_pred) / epsilon;
    }

    EXPECT_TRUE(H_num.isApprox(H_target, tolerance)) << "Target:" << std::endl
                                                     << H_target << std::endl
                                                     << "Actual:" << H_num << std::endl;
}

TEST(PredictAccelMeasurement, IdentityTransformationMakesMeasurementEqualState) {
    Eigen::Vector3d a(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d w(1.11, 2.22, 3.33);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();

    Eigen::Matrix<double, 3, 1> z_pred = predictAccelMeasurement(a, w, tf);

    EXPECT_TRUE(z_pred.isApprox(a, 1e-6))
        << "Target: " << a.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictAccelMeasurement, GravityAddedFromOrientation) {
    double g = 9.81;
    Eigen::Vector3d a(0, 0, 0);
    Eigen::Vector3d w(0, 0, 0);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();

    Eigen::Quaterniond orientation =
        (Eigen::Quaterniond)Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX());

    Eigen::Matrix<double, 3, 1> z_pred = predictAccelMeasurement(a, w, tf, true, orientation, g);

    Eigen::Matrix<double, 3, 1> z_target = orientation * Eigen::Vector3d(0, 0, g);

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-6))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictAccelMeasurement, StationaryZeroOrientationWithGravity) {
    double g = 9.81;
    Eigen::Vector3d a(0, 0, 0);
    Eigen::Vector3d w(0, 0, 0);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    Eigen::Quaterniond orientation(1, 0, 0, 0);

    Eigen::Matrix<double, 3, 1> z_pred = predictAccelMeasurement(a, w, tf, true, orientation, g);

    Eigen::Matrix<double, 3, 1> z_target(0, 0, g);

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-6))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictAccelMeasurement, LinearRatesOrientatedWithTf) {
    Eigen::Vector3d a(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d w(0, 0, 0);

    // Use a transform with zero translation to make sure no centripetal acceleration components
    // are added
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()));

    Eigen::Matrix<double, 3, 1> z_pred = predictAccelMeasurement(a, w, tf);

    Eigen::Matrix<double, 3, 1> z_target = tf.linear() * a;

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-6))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictAccelMeasurement, CentripetalComponentAddedToAcceleration) {
    // Zero the linear acceleration. Set the angular velocity to only be about one axis, and have
    // a translational offset on the other two axis. This will make it simple to compute the
    // expected accelerations.
    Eigen::Vector3d a(0, 0, 0);
    Eigen::Vector3d w(0, 0, 1);

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << 1, 2, 0;

    Eigen::Vector3d a_target;
    a_target.x() = -w.z() * w.z() * tf.translation().x();
    a_target.y() = -w.z() * w.z() * tf.translation().y();
    a_target.z() = 0.0;

    Eigen::Matrix<double, 3, 1> z_pred = predictAccelMeasurement(a, w, tf);

    EXPECT_TRUE(z_pred.isApprox(a_target, 1e-6))
        << "Target: " << a_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictGyroMeasurement, IdentityTransformationMakesMeasurementEqualState) {
    Eigen::Vector3d w(1.11, 2.22, 3.33);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();

    Eigen::Matrix<double, 3, 1> z_pred = predictGyroMeasurement(w, tf.linear());

    EXPECT_TRUE(z_pred.isApprox(w, 1e-6))
        << "Target: " << w.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(PredictGyroMeasurement, AngularRatesOrientatedWithTf) {
    Eigen::Vector3d w(1.11, 2.22, 3.33);

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()));

    Eigen::Matrix<double, 3, 1> z_pred = predictGyroMeasurement(w, tf.linear());

    Eigen::Matrix<double, 3, 1> z_target = tf.linear() * w;

    EXPECT_TRUE(z_pred.isApprox(z_target, 1e-6))
        << "Target: " << z_target.transpose() << ", Actual: " << z_pred.transpose();
}

TEST(AccelMeasurementJacobian, MatchesNumericalApproximationWithoutGravity) {
    accelJacobianMatchesNumericalApproximation(false);
}

TEST(AccelMeasurementJacobian, MatchesNumericalApproximationWithGravity) {
    accelJacobianMatchesNumericalApproximation(false);
}
