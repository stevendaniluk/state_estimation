#pragma once

#include <state_estimation/system_models/system_model.h>
#include <Eigen/Core>

namespace state_estimation {
namespace system_models {

// LinearSystemModel
//
// Describes linear dynamics of a system for modeling state transitions probabilities for state
// estimation algorithms.
//
// The system is described by:
//   i)  x' = Ax + Bu: State transition function
//   ii) R: Process covariance (can be time varying)
//
// The update method is responsible for updating all time varying matrix elements, if necessary.
class LinearSystemModel : public SystemModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    // @param m: Control dimensions
    LinearSystemModel(uint32_t n, uint32_t m);

    // A
    //
    // @return: State transition matrix
    Eigen::MatrixXd A() const;

    // setA
    //
    // @param new_A: New A matrix to use
    void setA(const Eigen::MatrixXd& new_A);

    // B
    //
    // @return: Control matrix
    Eigen::MatrixXd B() const;

    // setB
    //
    // @param new_B: New B matrix to use
    void setB(const Eigen::MatrixXd& new_B);

  protected:
    // State transition matrix
    Eigen::MatrixXd A_;
    // Control matrix
    Eigen::MatrixXd B_;
};

}  // namespace system_models
}  // namespace state_estimation
