#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/measurement_models/planer_2d_imu.h>
#include <state_estimation/utilities/imu_utilities.h>
#include <state_estimation/utilities/transformation_utilities.h>

namespace state_estimation {
namespace measurement_models {

Planer2DImu::Planer2DImu(bool compute_jacobian, bool include_gravity, double gravity)
    : NonlinearMeasurementModel::NonlinearMeasurementModel(planer_2d::state::DIMS, meas::imu::DIMS,
                                                           compute_jacobian)
    , include_gravity_(include_gravity)
    , gravity_(gravity) {}

void Planer2DImu::myUpdate(const Eigen::VectorXd& x) {
    // Form 3d versions of our state info
    Eigen::Vector3d a(x(planer_2d::state::AX), x(planer_2d::state::AY), 0);
    Eigen::Vector3d w(0, 0, x(planer_2d::state::VPSI));

    // Get the predicted measurement
    if (include_gravity_) {
        const Eigen::Quaterniond orientation = Eigen::Quaterniond(
            Eigen::AngleAxisd(x(planer_2d::state::PSI), Eigen::Vector3d::UnitZ()));
        z_pred_ = predictImuMeasurement(a, w, tf_, include_gravity_, orientation, gravity_);
    } else {
        z_pred_ = predictImuMeasurement(a, w, tf_);
    }

    if (compute_jacobian_) {
        // Compute the full Jacobian for an IMU. We'll only need a subset of it though since our
        // state is 2D. The rows of the IMU jacobian correspond to
        // [AX, AY, AZ, VPHI, VTHETA, VPSI], while the columns correspond to
        // [AX, AY, AZ, PHI, THETA, PSI, VPHI, VTHETA, VPSI].
        Eigen::Vector3d rpy(0, 0, x(planer_2d::state::PSI));
        Eigen::Matrix<double, 6, 9> H_6D =
            imuMeasurementJacobian(w, tf_, include_gravity_, rpy, gravity_);

        // Update the rows of the Jacobian for the measurement accelerations
        H_.block<3, 2>(meas::imu::AX, planer_2d::state::AX) = H_6D.block<3, 2>(0, 0);
        H_.block<3, 1>(meas::imu::AX, planer_2d::state::PSI) = H_6D.block<3, 1>(0, 5);
        H_.block<3, 1>(meas::imu::AX, planer_2d::state::VPSI) = H_6D.block<3, 1>(0, 8);

        // Update the rows of the Jacobian for the measurement angular rates
        H_.block<3, 2>(meas::imu::VPHI, planer_2d::state::AX) = H_6D.block<3, 2>(3, 0);
        H_.block<3, 1>(meas::imu::VPHI, planer_2d::state::PSI) = H_6D.block<3, 1>(3, 5);
        H_.block<3, 1>(meas::imu::VPHI, planer_2d::state::VPSI) = H_6D.block<3, 1>(3, 8);
    }
}

void Planer2DImu::postQUpdate() {
    // TODO
}

}  // namespace measurement_models
}  // namespace state_estimation
