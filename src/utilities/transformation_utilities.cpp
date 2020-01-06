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

}  // namespace state_estimation
