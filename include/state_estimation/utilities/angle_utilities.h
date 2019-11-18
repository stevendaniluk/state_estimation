#pragma once

namespace state_estimation {

// constrainAngle
//
//
// @param angle: Angle to constraint [rad]
// @return: Angle in the range [-PI, PI]
inline double constrainAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2 * M_PI;
    }

    return angle;
}

}  // namespace state_estimation
