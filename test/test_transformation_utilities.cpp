#include <gtest/gtest.h>
#include <state_estimation/utilities/transformation_utilities.h>

using namespace state_estimation;

TEST(LinearAccelerationTransformations, IdentityTransformDoesNotModifyAcceleration) {
    Eigen::Vector3d a(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d a_result =
        transformLinearAcceleration(a, Eigen::Vector3d::Zero(), Eigen::Isometry3d::Identity());

    EXPECT_TRUE(a_result.isApprox(a, 1e-6))
        << "Target: " << a.transpose() << ", Actual: " << a_result.transpose();
}

TEST(LinearAccelerationTransformations, OrientatedWithTf) {
    // Make a transform with a 45 degree rotation about each axis
    Eigen::Isometry3d tf;
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()));

    Eigen::Vector3d a(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d a_target = tf.linear() * a;

    Eigen::Vector3d a_result = transformLinearAcceleration(a, Eigen::Vector3d::Zero(), tf);

    EXPECT_TRUE(a_result.isApprox(a_target, 1e-6))
        << "Target: " << a_target.transpose() << ", Actual: " << a_result.transpose();
}

TEST(LinearAccelerationTransformations, CentripetalComponentAddedToAcceleration) {
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

    Eigen::Vector3d a_result = transformLinearAcceleration(a, w, tf);

    EXPECT_TRUE(a_result.isApprox(a_target, 1e-6))
        << "Target: " << a_target.transpose() << ", Actual: " << a_result.transpose();
}

TEST(LinearVelocityTransformations, IdentityTransformDoesNotModifyVelocity) {
    Eigen::Vector3d v(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d w(11, 22, 33);
    Eigen::Vector3d v_result = transformLinearVelocity(v, w, Eigen::Isometry3d::Identity());

    EXPECT_TRUE(v_result.isApprox(v, 1e-6))
        << "Target: " << v.transpose() << ", Actual: " << v_result.transpose();
}

TEST(LinearVelocityTransformations, ZeroTranslationOnlyRotatesVelocity) {
    Eigen::Vector3d v(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d w(11, 22, 33);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()));

    Eigen::Vector3d v_target = tf.linear() * v;

    Eigen::Vector3d v_result = transformLinearVelocity(v, w, tf);

    EXPECT_TRUE(v_result.isApprox(v_target, 1e-6))
        << "Target: " << v_target.transpose() << ", Actual: " << v_result.transpose();
}

TEST(LinearVelocityTransformations, TangentialVelocityAdded) {
    Eigen::Vector3d v(3.14159, 2.71828, 1.61803);
    Eigen::Vector3d w(11, 22, 33);
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << 1, 2, 3;

    Eigen::Vector3d v_target = v + w.cross(tf.translation());

    Eigen::Vector3d v_result = transformLinearVelocity(v, w, tf);

    EXPECT_TRUE(v_result.isApprox(v_target, 1e-6))
        << "Target: " << v_target.transpose() << ", Actual: " << v_result.transpose();
}

TEST(AngularVelocityTransformations, IdentityRotationDoesNotModifyVelocity) {
    Eigen::Vector3d w_target(3.14159, 2.71828, 1.61803);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Eigen::Vector3d w_result = transformAngularVelocity(w_target, R);
    EXPECT_TRUE(w_result.isApprox(w_target, 1e-6))
        << "Target: " << w_target.transpose() << ", Actual: " << w_result.transpose();
}

TEST(CovarianceTransformations, IdentityRotationDoesNotModifyVelocity) {
    Eigen::Matrix3d sigma_target;
    sigma_target << 1.1, 2.2, 3.3, 1.11, 2.22, 3.33, 1.111, 2.222, 3.333;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d sigma_result = transformCovariance(sigma_target, R);
    EXPECT_TRUE(sigma_result.isApprox(sigma_target, 1e-6))
        << "Target: " << std::endl
        << sigma_target.transpose() << ", Actual:" << std::endl
        << sigma_result.transpose();
}

TEST(CovarianceTransformations, ComputedWithRSigmaRt) {
    Eigen::Matrix3d sigma;
    sigma << 1.1, 2.2, 3.3, 1.11, 2.22, 3.33, 1.111, 2.222, 3.333;

    Eigen::Matrix3d R =
        Eigen::Matrix3d(Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()));

    Eigen::Matrix3d sigma_result = transformCovariance(sigma, R);
    Eigen::Matrix3d sigma_target = R * sigma * R.transpose();

    EXPECT_TRUE(sigma_result.isApprox(sigma_target, 1e-6))
        << "Target: " << std::endl
        << sigma_target.transpose() << ", Actual:" << std::endl
        << sigma_result.transpose();
}
