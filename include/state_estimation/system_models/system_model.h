#pragma once

#include <state_estimation/filters/filter_model.h>
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
//
// The transform stored by this object represents the transform from the control frame to the
// state frame.
class SystemModel : public FilterModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param m: Control dimensions
    // @param p: Process noise dimensions
    SystemModel(uint16_t n, uint16_t m, uint16_t p);

    // update
    //
    // Updates the system model about the current state and control input.
    //
    // @param x: Current system state
    // @param u: Control input
    // @param dt: Duration to step forward in time with the control
    void update(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt);

    // controlSize
    //
    // @return: Dimension of the control vector
    uint16_t controlSize() const;

    // activeControlSize
    //
    // @return: Dimensions of the control vector being usedupdated
    uint16_t activeControlSize() const;

    // activeControls
    //
    // @return: Ordered list of control indices being used
    std::vector<uint16_t> activeControls() const;

    // controlUsage
    //
    // @return: A bit field of which control variables are used (0=not used)
    std::vector<uint8_t> controlUsage() const;

    // setActiveControls
    //
    // @param active_controls: Which control variables to use (empty uses all variables)
    void setActiveControls(const std::vector<uint16_t>& active_controls);

    // setProcessCovariance
    //
    // @param R_p: The process covariance matrix to use in the covariance update computation
    //             R = P * R_p * P' + C * R_c * C'
    void setProcessCovariance(const Eigen::MatrixXd& R_p);

    // setControlCovariance
    //
    // @param R_c: The control covariance matrix to use in the covariance update computation
    //             R = P * R_p * P' + C * R_c * C'
    void setControlCovariance(const Eigen::MatrixXd& R_c);

    // Rp
    //
    // @return: Process covariance matrix
    Eigen::MatrixXd Rp() const;

    // Rc
    //
    // @return: Control covariance matrix
    Eigen::MatrixXd Rc() const;

    // P
    //
    // @return: Process noise Jacobian
    Eigen::MatrixXd P() const;

    // V
    //
    // @return: Control noise Jacobian
    Eigen::MatrixXd V() const;

  protected:
    // Internal implementations of update() and updateNoControl() to be defined by derived classes.
    virtual void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) = 0;

    // Dimension of the control vector
    uint16_t control_dims_;
    // Which control variables are being used (ordered)
    std::vector<uint16_t> active_controls_;
    // A bit field of which control variables are used (0=not used)
    std::vector<uint8_t> control_usage_;
    // Process noise covariance matrix
    Eigen::MatrixXd R_p_;
    // Control space covariance matrix
    Eigen::MatrixXd R_c_;
    // Process noise Jacobian
    Eigen::MatrixXd P_;
    // Control noise Jacobian
    Eigen::MatrixXd V_;
};

}  // namespace state_estimation
