#pragma once

namespace state_estimation {

// Defines the vector indices used for planer 2d states, controls, and measurements

namespace planer_2d {

// State vector indices.
//
// It is common for X, Y and PSI to correspond to the fixed frame, and all rates and accelerations
// to correspond to the inertial frame.
namespace state {
    static const int X = 0;
    static const int Y = 1;
    static const int VX = 2;
    static const int VY = 3;
    static const int AX = 4;
    static const int AY = 5;
    static const int PSI = 6;
    static const int VPSI = 7;

    static const int DIMS = 8;
}// end state namespace

}// end planer_2d namespace

} // end state_estimation namespace
