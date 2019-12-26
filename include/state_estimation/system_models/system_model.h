#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace state_estimation {

// SystemModel
//
// Base class for defining the interface for system models.
//
// Since many operations may be needed for both state prediction and covariance updates, this
// provides an update method so that all computations can be performed once in the same place, and
// the values (e.g. predicted state and covariance) can simply be retrieved with accessors.
//
// There is also a version of update that does not take in any control, for systems that aren't
// modeled with any control inputs and the current state should be propegated forward.
class SystemModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param m: Control dimentions
    SystemModel(uint32_t n, uint32_t m);

    // update
    //
    // Updates the system model about the current state and control input.
    //
    // @param x: Current system state
    // @param u: Control input
    // @param dt: Duration to step forward in time with the control
    void update(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt);

    // updateNoControl
    //
    // Updates the system model about the current state by simply propegating the state forward,
    // without applying a control
    //
    // @param x: Current system state
    // @param dt: Duration to propegate the state forward in time
    virtual void updateNoControl(const Eigen::VectorXd& x, double dt);

    // R
    //
    // @return Process covariance
    Eigen::MatrixXd R() const;

    // setR
    //
    // @param new_R: New process covariance to use
    void setR(const Eigen::MatrixXd& new_R);

    // setTf
    //
    // @param tf: Transformation from the control frame to the state frame
    void setTf(const Eigen::Isometry3d& tf);

    // stateSize
    //
    // @return: Number of state variables
    uint32_t stateSize() const;

    // controlSize
    //
    // @return: Number of control variables
    uint32_t controlSize() const;

  protected:
    // Internal implementations of update() and updateNoControl() to be defined by derived classes.
    virtual void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) = 0;
    virtual void myUpdateNoControl(const Eigen::VectorXd& x, double dt) = 0;

    // Dimension of the state vector
    uint32_t state_dims_;
    // Dimension of the control vector
    uint32_t control_dims_;
    // Process covariance
    Eigen::MatrixXd R_;
    // Transform from the control frame to the state frame
    Eigen::Isometry3d tf_;
};

}  // namespace state_estimation
