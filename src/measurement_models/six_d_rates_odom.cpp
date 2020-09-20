#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/measurement_models/six_d_rates_odom.h>
#include <state_estimation/utilities/math_utilities.h>
#include <state_estimation/utilities/transformation_utilities.h>

namespace state_estimation {
namespace measurement_models {

using namespace six_d_rates;

SixDRatesOdom::SixDRatesOdom(bool compute_jacobian)
    : NonlinearMeasurementModel::NonlinearMeasurementModel(state::DIMS, 4, compute_jacobian) {
    // Call this so that the Jacobian gets updated for the default identity transformation
    postTfUpdate();
}

void SixDRatesOdom::myUpdate(const Eigen::VectorXd& x, double dt) {
    z_pred_.segment(meas::odom::VX, 3) =
        transformLinearVelocity(x.segment(state::VX, 3), x.segment(state::VPHI, 3), tf_);
    z_pred_(meas::odom::VPSI) = transformAngularVelocity(x.segment(state::VPHI, 3), tf_.linear()).z();

    // Jacobian doesn't change with time
}

void SixDRatesOdom::postTfUpdate() {
    if (compute_jacobian_) {
        // The state linear and angular velocities are transformed to the measurement frame, the
        // jacobian accounts for that transformation

        // Measured linear velocity with respect to state linear velocity
        H_.block<3, 3>(meas::odom::VX, state::VX) = tf_.linear();

        // Measured linear velocity with respect to state angular velocity
        H_.block<3, 3>(meas::odom::VX, state::VPHI) = -skew(tf_.translation()) * tf_.linear();

        // Measured angular velocity with respect to state angular velocity
        H_.block<1, 3>(meas::odom::VPSI, state::VPHI) = tf_.linear().row(2);
    }
}

}  // namespace measurement_models
}  // namespace state_estimation
