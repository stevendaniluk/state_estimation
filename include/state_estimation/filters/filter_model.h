#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace state_estimation {

// FilterModel
//
// Base class for containing common functionality in system and measurement models.
//
// The model is responsible for defining arithmetic operations for state/measurement vectors
// (addition, subtraction, and weighted sum).
class FilterModel {
  public:
    // Constructor
    //
    // @param n: State dimensions
    FilterModel(uint16_t n);

    // setTf
    //
    // @param tf: Transformation
    void setTf(const Eigen::Isometry3d& tf);

    // stateSize
    //
    // @return: Dimenions of the full state vector
    uint16_t stateSize() const;

    // activeStateSize
    //
    // @return: Dimensions of the state vector being updated
    uint16_t activeStateSize() const;

    // activeStates
    //
    // @return: Ordered list of state indices being updated
    std::vector<uint16_t> activeStates() const;

    // stateUsage
    //
    // @return: A bit field of which state variables are used (0=not used)
    std::vector<uint8_t> stateUsage() const;

    // setActiveStates
    //
    // @param active_states: Which state variables to update (empty updates all variables)
    void setActiveStates(const std::vector<uint16_t>& active_states);

    // setCheckStationary
    //
    // @param check: When true, the system will be checked for being stationary on each update
    void setCheckStationary(bool check);

    // checkStationary
    //
    // @return: True when this model will evaluate the input to check for stationary conditions
    bool checkStationary();

    // isStationary
    //
    // @param x: State to evaluate
    // @param data: Control/Measurement to evaluate
    // @return: True when the state and control/measurement is considered stationary
    bool isStationary(const Eigen::VectorXd& x, const Eigen::VectorXd& data) const;

    // makeStationary
    //
    // @param x: State to modify to be stationary
    // @param cov: Covariance to modify to be stationary
    void makeStationary(Eigen::VectorXd* x, Eigen::MatrixXd* cov) const;

    // setIsStationaryFunction
    //
    // @param f: Function to envoke when isStationary() is called
    void setIsStationaryFunction(
        const std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)>& f);

    // setMakeStationaryFunction
    //
    // @param f: Function to envoke when isStationary() is called
    void setMakeStationaryFunction(
        const std::function<void(Eigen::VectorXd*, Eigen::MatrixXd*)>& f);

    // processStationaryInput
    //
    // Optional method to perform some action upon receiving a stationary state (e.g. updating
    // biases).
    //
    // @param x: Stationary state
    // @param data: Control/Measurement
    virtual void processStationaryInput(const Eigen::VectorXd& x, const Eigen::VectorXd& data) {}

    // addVectors
    //
    // Provides the addition operation for two vectors. This is by default simply lhs + rhs,
    // but can be overridden to apply additional operations like normalization or constraints.
    //
    // @param lhs: Left side vector
    // @param rhs: Right side vector
    // @return: lhs + rhs
    virtual Eigen::VectorXd addVectors(const Eigen::VectorXd& lhs,
                                       const Eigen::VectorXd& rhs) const;

    // subtractVectors
    //
    // Provides the subtraction operation for two vectors. This is by default simply lhs - rhs,
    // but can be overridden to apply additional operations like normalization or constraints.
    //
    // @param lhs: Left side vector
    // @param rhs: Right side vector
    // @return: lhs - rhs
    virtual Eigen::VectorXd subtractVectors(const Eigen::VectorXd& lhs,
                                            const Eigen::VectorXd& rhs) const;

    // weightedSum
    //
    // Provides the operation to compute the weighted sum of a set of vectors, i.e.
    //   X' = sum_i(w * X_i)
    //
    // @param w: Scalar weight values
    // @param X: Matrix of data to process, each column is a vector to scale by the weight
    // @return: Weighted sum
    virtual Eigen::VectorXd weightedSum(const Eigen::VectorXd& w, const Eigen::MatrixXd& X) const;

  protected:
    // postTfUpdate
    //
    // Optional operation to be performed when the transformation for control/measurement inputs
    // changes.
    virtual void postTfUpdate() {}

    // Dimension of the state vector
    uint16_t state_dims_;
    // Dimensions of the state vector being used
    uint16_t active_state_dims_;
    // Which state variables are being updated (ordered)
    std::vector<uint16_t> active_states_;
    // A bit field of which states are used (0=not used)
    std::vector<uint8_t> state_usage_;
    // If checks should be performed for determining if the system is stationary
    bool check_stationary_;
    // Transformation for control/measurement inputs
    Eigen::Isometry3d tf_;
    // Flag for if a non identity transformation has been provided
    bool tf_set_;

  private:
    // Function for determining if a state and control/measurement input is stationary
    std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)> is_stationary_f_;
    // Function for modifying a state vector and covariance matrix when stationary
    std::function<void(Eigen::VectorXd*, Eigen::MatrixXd*)> make_stationary_f_;
};

}  // namespace state_estimation
