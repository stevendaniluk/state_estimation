#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/planer_2d_motion.h>
#include <state_estimation/measurement_models/planer_2d_nh_odometry.h>

namespace state_estimation {
namespace measurement_models {

using namespace planer_2d;

Planer2DNhOdometry::Planer2DNhOdometry(bool compute_jacobian)
    : NonlinearMeasurementModel::NonlinearMeasurementModel(state::DIMS, meas::nh_odom::DIMS,
                                                           compute_jacobian) {
    // We're directly observing linear and angular velocity so these entries in the Jacobian are
    // unity
    H_(meas::nh_odom::VX, state::VX) = 1;
    H_(meas::nh_odom::VPSI, state::VPSI) = 1;
}

void Planer2DNhOdometry::myUpdate(const Eigen::VectorXd& x) {
    z_pred_(meas::nh_odom::VX) = x(state::VX);
    z_pred_(meas::nh_odom::VPSI) = x(state::VPSI);
}

}  // namespace measurement_models
}  // namespace state_estimation
