#include <state_estimation/utilities/imu_utilities.h>
#include <state_estimation/utilities/transformation_utilities.h>

namespace state_estimation {

Eigen::Matrix<double, 6, 1> predictImuMeasurement(const Eigen::Vector3d& a,
                                                  const Eigen::Vector3d& w,
                                                  const Eigen::Isometry3d tf, bool include_gravity,
                                                  Eigen::Quaterniond orientation, double g) {
    Eigen::Matrix<double, 6, 1> z;

    if (include_gravity) {
        z.segment<3>(0) =
            transformLinearAcceleration(a + orientation * Eigen::Vector3d(0, 0, g), w, tf);
    } else {
        z.segment<3>(0) = transformLinearAcceleration(a, w, tf);
    }
    z.segment<3>(3) = transformAngularVelocity(w, tf.linear());

    return z;
}

Eigen::Matrix<double, 6, 9> imuMeasurementJacobian(const Eigen::Vector3d& w,
                                                   const Eigen::Isometry3d& tf,
                                                   bool include_gravity, Eigen::Vector3d rpy,
                                                   double g) {
    // Many entries in the Jacobian will be zero
    Eigen::Matrix<double, 6, 9> H = Eigen::Matrix<double, 6, 9>::Zero();

    const Eigen::Matrix3d& R = tf.linear();
    const Eigen::Vector3d& t = tf.translation();

    // Change in measured acceleration with respect to state acceleration is the rotation matrix
    // between the frames
    H.block<3, 3>(0, 0) = R;

    // Change in measured angular rates with respect to state angular rates is the rotation matrix
    // between the frames
    H.block<3, 3>(3, 6) = R;

    // The change in measured acceleration with respect to state angular rates depends on the
    // translational offset between the frames and the angular rate
    const Eigen::Vector3d i = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d j = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d k = Eigen::Vector3d::UnitZ();

    H.block<3, 1>(0, 6) = R * (i.cross(w.cross(t)) + w.cross(i.cross(t)));
    H.block<3, 1>(0, 7) = R * (j.cross(w.cross(t)) + w.cross(j.cross(t)));
    H.block<3, 1>(0, 8) = R * (k.cross(w.cross(t)) + w.cross(k.cross(t)));

    if (include_gravity) {
        // The change in measured acceleration with respect to state orientation is due to the
        // effect of gravity. We'll need to compute the derivative of the state orientation with
        // respect to each euler angle
        const double sphi = sin(rpy(0));
        const double cphi = cos(rpy(0));
        const double stheta = sin(rpy(1));
        const double ctheta = cos(rpy(1));
        const double spsi = sin(rpy(2));
        const double cpsi = cos(rpy(2));

        Eigen::Vector3d dRs_dphi;
        dRs_dphi.x() = -sphi * stheta * cpsi + cphi * spsi;
        dRs_dphi.y() = -sphi * stheta * spsi - cphi * cpsi;
        dRs_dphi.z() = -sphi * ctheta;

        Eigen::Vector3d dRs_dtheta;
        dRs_dtheta.x() = cphi * ctheta * cpsi;
        dRs_dtheta.y() = cphi * ctheta * spsi;
        dRs_dtheta.z() = -cphi * stheta;

        Eigen::Vector3d dRs_dpsi;
        dRs_dpsi.x() = -cphi * stheta * spsi + sphi * cpsi;
        dRs_dpsi.y() = cphi * stheta * cpsi + sphi * spsi;
        dRs_dpsi.z() = 0;

        H.block<3, 1>(0, 3) = g * R * dRs_dphi;
        H.block<3, 1>(0, 4) = g * R * dRs_dtheta;
        H.block<3, 1>(0, 5) = g * R * dRs_dpsi;
    }

    return H;
}

}  // namespace state_estimation
