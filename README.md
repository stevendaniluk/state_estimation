# State Estimation

This is a verstaile state estimation library for Bayesian filters.

Features:
* KF, EKF, and UKF algorithms
* System and measurement models are independent of the algorithm (e.g. the same mesurement model can be used for an EKF and a UKF)
* Ability to rewind the filter state for measurements in the past and re apply updates
* Ability update only a subset of the state variables and use only a subset of the control or measurement vectors
* Ability to define and handle transormation between the state frame and the measurement frames
* Predicates for defining when the state should be updated and forcing the state variables to stationary values (e.g. estimating velocities on a vehicle and clamping them to zero based on some heuristic that the vehicle has stopped)

# Library Design

## Architecture
This library separates the system and measurement models, which are what truly define the state estimation problem, from the actual filter algorithms. Models are also defined as being linear or non-linear, where the linear models are only compatabile with the vanilla Kalman Filter algorithm. The state representation is also separate from the filters and models, this way multiple versions of a system or measurement model can be interchanged using the same state representation.

State definitions are defined in `include/state_estimation/definitions`. These also define operators for addition, subtraction, and weighted sums in the event that some state variables requires some post processing (e.g. angle wrap, normalization, etc.).

The models have the following heirachy:
* **FilterModel** - Base class for all models, defines common data such as state size and transformations between frames
   * **SystemModel** - Adds the notion of a control vector, process noise, and control noise to the base model
      * **LinearSystemModel** - Defines the **A** and **B** matrices for the update **x' = Ax + Bu**
      * **NonlinearSystemModel** - Defines the nonlinear state prediction function **x' = g(x, u)** and Jacobian **G**
   * **MeasurementModel** - Adds the notion of a measurement vector and measurement noise covariance to the base model
      * **LinearMeasurementModel** - Defines the observation matrix **C** in **y = Cx**
      * **NonlinearMeasurementModel** - Defines the nonlinear observation function **y = h(x)** and the Jacobian **H**

For a new application, one would define their own state representation, define the addition, subtraction, and weighted sum operators, then create their own system and measurement models deriving from either the linear or nonlinear varients.

There is one application already present, there is a state definition for estimated 6D rates (linear and angular velocity) using odometry and IMU measurements.

The filters have the following hierarchy:
* **FilterBase** - Handles all the bookkeeping of the state, covariance, predicitions, corrections, and any rewind operations that need to occur. The actual prediction and correction update steps are left to the derived classes to define.
   * **KalmanFilter** - Kalman Filter algorithm prediction and correction equations
      * **KalmanFilterVS** - Variant of KalmanFilter that allows using a subset of the state/control/measurement
  * **EKF** - Extended Kalman Filter algorithm prediction and correction equations
     * **EKFVS** - Variant of EKF that allows using a subset of the state/control/measurement
 * **UKF** - Unscented Kalman Filter algorithm prediction and correction equations
    * **UKFVS** - Variant of UKF that allows using a subset of the state/control/measurement

Any common math operations (e.g. RK4 integration) are kept in the `utilities` directory to promote reuse across models.

## Comments
This library was created with versatility in mind and was primarily intended for prototyping. The objective was to separate the filter algorithms from the processes they were operating on, and to enable reusing measurement and system models between applications.

A consequence of the versatility is that some design decisions, such as opting for runtime polymorphism over compile time, and using dynamic sized vectors/matrices over static sized, will incur a slight performance cost. For most applications this should be negligable. If that extra performance is really that important then you probably aren't iterating on designs so the versatility of this library is of little benefit anyways.

# Compiling and Running

To compile:
```bash
mkdir build
cd build
cmake ..
make
```

To run the unit tests:
```bash
cd build
ctest
```

**Note:** By default this will attempt to build the unit test by default, which requires Gtest. There are three options here:
1. Don't build unit tests by running `cmake .. -DBUILD_TESTING=OFF`
1. Have this project build Gtest by running `cmake .. -DSTATE_ESTIMATION_BUILD_GTEST=ON`
1. Build Gtest on your own so that it can be found with `find_package(GTest)`

A debug flag can also be defined to output debug logs, add the cmake option `-DDEBUG_STATE_ESTIMATION=ON`. **Warning:** this will output a lot of logs, it prints the full state, measurement, covariance, Jacobians, Kalman gain, and innovation each update.

For an example how to use and run the filter check out the examples directory.

# Building As A ROS Package
To use this as a ROS package and have catkin build it you can create a simple dummy catkin package that adds this repo as a subdirectory.

Make a new package folder and clone this repo in it
```bash
mkdir ~/my_worspace/state_estimation
cd ~/my_worspace/state_estimation
git clone git@github.com:stevendaniluk/state_estimation.git
```

Then make a `CMakeLists.txt` file with:
```
cmake_minimum_required(VERSION 2.8.3)
project(state_estimation)

add_subdirectory(state_estimation)

catkin_package(
  INCLUDE_DIRS ${STATE_ESTIMATION_INCLUDE_DIRS}
  LIBRARIES state_estimation
)
```

And a `package.xml` file with:
```xml
<?xml version="1.0"?>
<package>
  <name>state_estimation</name>
  <version>0.0.0</version>
  <description>TODO</description>

  <maintainer email="someone@something.com">Someone</maintainer>
  <author email="someone@something.com">Someone</author>
  <license>MIT</license>

  <buildtool_depend>catkin</buildtool_depend>

</package>
```

You can then add the `state_estimation` catkin package as a build dependency for any other ROS packages, such as [state_estimation_ros](https://github.com/stevendaniluk/state_estimation_ros).
