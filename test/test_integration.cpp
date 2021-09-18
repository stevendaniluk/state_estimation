#include <gtest/gtest.h>
#include <state_estimation/utilities/integration.h>

using namespace state_estimation;

void quaternionEqual(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2,
                     double epsilon = 1e-6) {
    EXPECT_NEAR(q1.w(), q2.w(), epsilon);
    EXPECT_NEAR(q1.x(), q2.x(), epsilon);
    EXPECT_NEAR(q1.y(), q2.y(), epsilon);
    EXPECT_NEAR(q1.z(), q2.z(), epsilon);
}

// getStraightLineAccelerationProfile
//
// Generates position, velocity, and time samples from a body moving along a straight path with
// a constant acceleration. The sample points will be evenly spaced along the path with respect
// to time, starting at [0, 0] with the velocity aligned in the +ve x direction.
//
// This is the solution to the ODE:
//   [p'] = [R(t) * v]
//   [v'] = [R(t) * a]
//
// which provides source of ground truth for evaluating position and velocity integration.
//
// @param vi: Intiial velocity
// @param a_mag: Acceleration magnitude
// @param duration: Duration to generate profile for
// @param n: Number of steps
// @param p: Position points to be computed
// @param v: linear velocity vectors to populate
// @param w: Angular velocity vectors to populate
// @param a: Linear acceleration vectors to populate
// @param t: Time stamps (starting from 0)
void getStraightLineAccelerationProfile(Eigen::Vector3d vi, Eigen::Vector3d a_mag, double duration,
                                        size_t n, std::vector<Eigen::Vector3d>* p,
                                        std::vector<Eigen::Vector3d>* v, std::vector<double>* t) {
    // Form a profile with a constant acceleration. The positions are evenly spaced with respect to
    // time, and the velocity is computed from the distance travelled under a constant acceleration.
    p->resize(n);
    v->resize(n);
    t->resize(n);

    for (int i = 0; i < n; ++i) {
        double dt = duration * (static_cast<double>(i) / (n - 1));

        (*t)[i] = dt;
        (*p)[i] = vi * dt + 0.5 * pow(dt, 2) * a_mag;
        (*v)[i] = vi + dt * a_mag;
    }
}

// getCircleAccelerationProfile
//
// Generates position, velocity, and time samples from a body moving along a circlular path with
// a constant acceleration. The sample points will be evenly spaced along the circle with respect
// to time, starting at [0, -r] with the velocity aligned in the +ve x direction.
//
// This is the solution to the ODE:
//   [p'] = [R(t) * v]
//   [v'] = [R(t) * a]
//
// which provides source of ground truth for evaluating position and velocity integration.
//
// @param vi: Intiial velocity
// @param a_mag: Acceleration magnitude
// @param duration: Duration to generate profile for
// @param n: Number of steps
// @param p: Position points to be computed
// @param v: linear velocity vectors to populate
// @param w: Angular velocity vectors to populate
// @param a: Linear acceleration vectors to populate
// @param t: Time stamps (starting from 0)
void getCircleAccelerationProfile(double vi, double a_mag, double duration, size_t n, double r,
                                  std::vector<Eigen::Vector3d>* p, std::vector<Eigen::Vector3d>* v,
                                  std::vector<Eigen::Vector3d>* w, std::vector<Eigen::Vector3d>* a,
                                  std::vector<double>* t) {
    // Form a profile that accelerates along a complete circle with a constant acceleration. The
    // positions are evenly spaced with respect to time, and the velocity is computed from the arc
    // length travelled under a constant acceleration.
    p->resize(n);
    v->resize(n);
    w->resize(n);
    a->resize(n);
    t->resize(n);

    for (int i = 0; i < n; ++i) {
        // Step distance, arc length, and angle
        double dt = duration * (static_cast<double>(i) / (n - 1));
        double ds = vi * dt + 0.5 * a_mag * pow(dt, 2);
        double dtheta = ds / r;

        // Position
        (*p)[i].x() = std::cos(dtheta - M_PI / 2) * r;
        (*p)[i].y() = std::sin(dtheta - M_PI / 2) * r;
        (*p)[i].z() = 0.0;

        // Linear velocity from constant acceleration, set as tangent to the circle
        double v_new = vi + a_mag * dt;
        (*v)[i].x() = std::cos(dtheta) * v_new;
        (*v)[i].y() = std::sin(dtheta) * v_new;
        (*v)[i].z() = 0;

        // Angular velocity
        (*w)[i].x() = 0.0;
        (*w)[i].y() = 0.0;
        (*w)[i].z() = v_new / r;

        // Linear acceleration, sum of tangential and centripetal
        Eigen::Vector3d a_cent = (*w)[i].cross((*v)[i]);
        Eigen::Vector3d a_tang(a_mag * std::cos(dtheta), a_mag * std::sin(dtheta), 0.0);
        (*a)[i] = a_cent + a_tang;

        // Time
        (*t)[i] = dt;
    }
}

TEST(Integration, ZeroAngularRateProducesZeroDeltaQuaternion) {
    double dt = 100.0;
    Eigen::Vector3d w(0.0, 0.0, 0.0);
    Eigen::Quaterniond q = deltaQuaternion(dt, w);

    EXPECT_FLOAT_EQ(q.w(), 1.0);
    EXPECT_FLOAT_EQ(q.x(), 0.0);
    EXPECT_FLOAT_EQ(q.y(), 0.0);
    EXPECT_FLOAT_EQ(q.z(), 0.0);
}

TEST(Integration, DeltaQuaternionEqualBothDirections) {
    double dt = 0.123;
    Eigen::Vector3d w(1.1, 2.2, 3.3);

    Eigen::Quaterniond q = deltaQuaternion(dt, w);
    Eigen::Quaterniond q_rev = deltaQuaternion(dt, -w);

    EXPECT_FLOAT_EQ(q.w(), q_rev.w());
    EXPECT_FLOAT_EQ(q.x(), -q_rev.x());
    EXPECT_FLOAT_EQ(q.y(), -q_rev.y());
    EXPECT_FLOAT_EQ(q.z(), -q_rev.z());
}

TEST(Integration, DeltaQuaternionKnownValue) {
    double dt = 0.1;

    Eigen::Vector3d w_x(1.1, 0.0, 0.0);
    Eigen::Quaterniond q_x = deltaQuaternion(dt, w_x);
    Eigen::Quaterniond q_x_ref =
        (Eigen::Quaterniond)Eigen::AngleAxisd(dt * w_x.x(), Eigen::Vector3d::UnitX());
    quaternionEqual(q_x_ref, q_x);

    Eigen::Vector3d w_y(0.0, 2.2, 0.0);
    Eigen::Quaterniond q_y = deltaQuaternion(dt, w_y);
    Eigen::Quaterniond q_y_ref =
        (Eigen::Quaterniond)Eigen::AngleAxisd(dt * w_y.y(), Eigen::Vector3d::UnitY());
    quaternionEqual(q_y_ref, q_y);

    Eigen::Vector3d w_z(0.0, 0.0, 3.3);
    Eigen::Quaterniond q_z = deltaQuaternion(dt, w_z);
    Eigen::Quaterniond q_z_ref =
        (Eigen::Quaterniond)Eigen::AngleAxisd(dt * w_z.z(), Eigen::Vector3d::UnitZ());
    quaternionEqual(q_z_ref, q_z);
}

TEST(Integration, StraightLineVelocityIntegration) {
    Eigen::Vector3d vi(1.111, 2.222, 3.333);
    Eigen::Vector3d a(3.14159, 2.71828, 1.41421);
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
    double dt = 0.2;
    size_t n = 21;

    std::vector<Eigen::Vector3d> p_ref;
    std::vector<Eigen::Vector3d> v_ref;
    std::vector<double> t_ref;

    getStraightLineAccelerationProfile(vi, a, dt, n, &p_ref, &v_ref, &t_ref);

    double v_epsilon = 0.01;
    Eigen::Vector3d v = v_ref[0];
    for (int i = 0; i < v_ref.size() - 1; ++i) {
        double dt = t_ref[i + 1] - t_ref[i];
        integrateVelocityRK4(dt, a, w, &v);

        double v_error = fabs(v.norm() - v_ref[i + 1].norm()) / v_ref[i + 1].norm();

        EXPECT_LT(v_error, v_epsilon)
            << "Step " << i << ", v=" << v.transpose() << ", v_ref=" << v_ref[i + 1].transpose();
    }
}

TEST(Integration, StraightLinePositionIntegration) {
    Eigen::Vector3d vi(1.111, 2.222, 3.333);
    Eigen::Vector3d a(3.14159, 2.71828, 1.41421);
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
    double dt = 0.2;
    size_t n = 21;

    std::vector<Eigen::Vector3d> p_ref;
    std::vector<Eigen::Vector3d> v_ref;
    std::vector<double> t_ref;

    getStraightLineAccelerationProfile(vi, a, dt, n, &p_ref, &v_ref, &t_ref);

    double p_epsilon = 0.01;
    Eigen::Vector3d p = p_ref[0];
    Eigen::Vector3d v = v_ref[0];
    for (int i = 0; i < p_ref.size() - 1; ++i) {
        double dt = t_ref[i + 1] - t_ref[i];
        const Eigen::Vector3d vi = v;
        integratePositionRK4(dt, vi, a, w, &p, &v);

        double p_error = fabs(p.norm() - p_ref[i + 1].norm()) / p_ref[i + 1].norm();

        EXPECT_LT(p_error, p_epsilon)
            << "Step " << i << ", p=" << p.transpose() << ", p_ref=" << p_ref[i + 1].transpose();
    }
}

TEST(Integration, AcceleratingCircleVelocityIntegration) {
    // Use the ground truth profile along a circle
    double a = 2.71828;
    double vi = 3.14159;
    double dt = 0.2;
    size_t n = 21;
    double r = 2.0;

    std::vector<Eigen::Vector3d> p_ref;
    std::vector<Eigen::Vector3d> v_ref;
    std::vector<Eigen::Vector3d> w_ref;
    std::vector<Eigen::Vector3d> a_ref;
    std::vector<double> t_ref;

    getCircleAccelerationProfile(vi, a, dt, n, r, &p_ref, &v_ref, &w_ref, &a_ref, &t_ref);

    // Make sure the norm and direction of the velocity are within some error percentage
    double norm_epsilon = 0.01;
    double theta_epsilon = 0.01;
    Eigen::Vector3d v = v_ref[0];
    for (int i = 0; i < v_ref.size() - 1; ++i) {
        double dt = t_ref[i + 1] - t_ref[i];
        integrateVelocityRK4(dt, a_ref[i], w_ref[i], &v);

        double norm_error = fabs(v.norm() - v_ref[i + 1].norm()) / v_ref[i + 1].norm();
        double theta_error = fabs(std::acos(v.normalized().dot(v_ref[i + 1].normalized())));

        EXPECT_LT(norm_error, norm_epsilon)
            << "Step " << i << ", v=" << v.transpose() << ", v_ref=" << v_ref[i + 1].transpose();
        EXPECT_LT(theta_error, theta_epsilon)
            << "Step " << i << ", v=" << v.transpose() << ", v_ref=" << v_ref[i + 1].transpose();
    }
}

TEST(Integration, AcceleratingCirclePositionIntegration) {
    // Use the ground truth profile along a circle
    double a = 2.71828;
    double vi = 3.14159;
    double dt = 0.2;
    size_t n = 21;
    double r = 2.0;

    std::vector<Eigen::Vector3d> p_ref;
    std::vector<Eigen::Vector3d> v_ref;
    std::vector<Eigen::Vector3d> w_ref;
    std::vector<Eigen::Vector3d> a_ref;
    std::vector<double> t_ref;

    getCircleAccelerationProfile(vi, a, dt, n, r, &p_ref, &v_ref, &w_ref, &a_ref, &t_ref);

    // Make sure the absolute positionis within some error percentage
    double epsilon = 0.01;
    Eigen::Vector3d p = p_ref[0];
    Eigen::Vector3d v = v_ref[0];
    for (int i = 0; i < v_ref.size() - 1; ++i) {
        double dt = t_ref[i + 1] - t_ref[i];
        integratePositionRK4(dt, v, a_ref[i], w_ref[i], &p, &v);

        double error = (p - p_ref[i + 1]).norm();

        EXPECT_LT(error, epsilon) << "Step " << i << ", p=" << p.transpose()
                                  << ", p_ref=" << p_ref[i + 1].transpose();
    }
}
