#pragma once

namespace state_estimation {

// Defines the vector indices used for 6D states, controls, and measurements

namespace six_dof {

// State vector indices.
//
// It is common for X, Y and PSI to correspond to the fixed frame, and all rates and accelerations
// to correspond to the inertial frame.
namespace state {
    static const int X = 0;
    static const int Y = 1;
    static const int Z = 2;
    static const int VX = 3;
    static const int VY = 4;
    static const int VZ = 5;
    static const int AX = 6;
    static const int AY = 7;
    static const int AZ = 8;
    static const int PHI = 9;
    static const int THETA = 10;
    static const int PSI = 11;
    static const int VPHI = 12;
    static const int VTHETA = 13;
    static const int VPSI = 14;

    static const int DIMS = 15;
}// end state namespace

}// end six_dof namespace

} // end state_estimation namespace
