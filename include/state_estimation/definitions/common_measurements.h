#pragma once

namespace state_estimation {

// Defines the vector indices used for measurements that are common to different state/motion types

namespace meas {

// Accelerometer
namespace accel {
    static const int AX = 0;
    static const int AY = 1;
    static const int AZ = 2;

    static const int DIMS = 3;
}

// Gyroscope
namespace gyro {
    static const int VPHI = 0;
    static const int VTHETA = 1;
    static const int VPSI = 2;

    static const int DIMS = 3;
}

// IMU (accelerometer + gyroscope)
namespace imu {
    static const int AX = 0;
    static const int AY = 1;
    static const int AZ = 2;
    static const int VPHI = 3;
    static const int VTHETA = 4;
    static const int VPSI = 5;

    static const int DIMS = 6;
}

// IMU with orientation in roll-pitch-yaw representation
namespace imu_rpy {
    static const int AX = 0;
    static const int AY = 1;
    static const int AZ = 2;
    static const int VPHI = 3;
    static const int VTHETA = 4;
    static const int VPSI = 5;
    static const int PHI = 6;
    static const int THETA = 7;
    static const int PSI = 8;

    static const int DIMS = 9;
}

// IMU with orientation in quaternion representation
namespace imu_quat {
    static const int AX = 0;
    static const int AY = 1;
    static const int AZ = 2;
    static const int VPHI = 3;
    static const int VTHETA = 4;
    static const int VPSI = 5;
    static const int W = 6;
    static const int X = 7;
    static const int Y = 8;
    static const int Z = 9;

    static const int DIMS = 10;
}

}// end meas namespace

} // end state_estimation namespace
