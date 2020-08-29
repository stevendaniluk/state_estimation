#include <state_estimation/utilities/math_utilities.h>

namespace state_estimation {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d skew;
    skew << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;

    return skew;
}

}  // namespace state_estimation
