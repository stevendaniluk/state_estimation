#include <state_estimation/utilities/transformation_utilities.h>

namespace state_estimation {

Eigen::Vector3d transformLinearAcceleration(const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                                            const Eigen::Isometry3d& tf) {
    return tf.linear() * (a + w.cross(w.cross(tf.translation())));
}

Eigen::Vector3d transformLinearVelocity(const Eigen::Vector3d& v, const Eigen::Vector3d& w,
                                        const Eigen::Isometry3d& tf) {
    return tf.linear() * v + (tf.linear() * w).cross(tf.translation());
}

Eigen::Vector3d transformAngularVelocity(const Eigen::Vector3d& w, const Eigen::Matrix3d& R) {
    return R * w;
}

Eigen::Matrix3d transformCovariance(const Eigen::Matrix3d& cov, const Eigen::Matrix3d& R) {
    return R * cov * R.transpose();
}

Eigen::Quaterniond orientationFromGravityVector(Eigen::Vector3d g) {
    // The orientation can be determined by solving:
    //   ||g|| = q * [0, 0, -1]' * q' (where q' is the conjugate of q)
    // And setting the z component of the quaternion q to zero
    g.normalize();
    const double g_norm_sq = g.squaredNorm();

    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (g_norm_sq > 1e-9) {
        q.w() = sqrt(0.5 * (-g.z() + sqrt(g_norm_sq)));
        q.x() = g.y() / (2 * q.w());
        q.y() = -g.x() / (2 * q.w());
        q.z() = 0;
    }

    return q;
}

}  // namespace state_estimation
