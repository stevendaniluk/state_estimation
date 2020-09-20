#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/measurement_models/six_d_rates_imu.h>

namespace state_estimation {
namespace measurement_models {

using namespace six_d_rates;

SixDRatesImu::SixDRatesImu(bool compute_jacobian)
    : NonlinearMeasurementModel::NonlinearMeasurementModel(state::DIMS, meas::imu::DIMS,
                                                           compute_jacobian) {
    // Call this so that the Jacobian gets updated for the default identity transformation
    postTfUpdate();
}

void SixDRatesImu::myUpdate(const Eigen::VectorXd& x, double dt) {
    z_pred_.segment(meas::imu::AX, 3) =
        x.segment(state::AX, 3) - x.segment(state::B_AX, 3) + x.segment(state::GX, 3);
    z_pred_.segment(meas::imu::VPHI, 3) = x.segment(state::VPHI, 3) - x.segment(state::B_WX, 3);

    if (tf_set_) {
        z_pred_.segment(meas::imu::AX, 3) = tf_.linear() * z_pred_.segment(meas::imu::AX, 3);
        z_pred_.segment(meas::imu::VPHI, 3) = tf_.linear() * z_pred_.segment(meas::imu::VPHI, 3);
    }

    // Jacobian doesn't change with time
}

void SixDRatesImu::postTfUpdate() {
    if (compute_jacobian_) {
        // Measured acceleration wrt state linear acceleration
        H_.block<3, 3>(meas::imu::AX, state::AX) = tf_.linear();

        // Measured linear acceleration wrt accelerometer baises
        H_.block<3, 3>(meas::imu::AX, state::B_AX) = -tf_.linear();

        // Measured linear acceleration wrt gravity
        H_.block<3, 3>(meas::imu::AX, state::GX) = tf_.linear();

        // Measured angular velocity wrt state angular velocity
        H_.block<3, 3>(meas::imu::VPHI, state::VPHI) = tf_.linear();

        // Measured angular velocity wrt gyroscope biases
        H_.block<3, 3>(meas::imu::VPHI, state::B_WX) = -tf_.linear();
    }
}

}  // namespace measurement_models
}  // namespace state_estimation
