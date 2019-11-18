#pragma once

#include <Eigen/Core>
#include <deque>
#include <iostream>
#include <vector>

namespace state_estimation {

// FilterBase
//
// Provides common functionality for Bayesian style state estimation filters.
//
// This class manages:
//   -Maintaining the internal filter state
//   -Maintaining a history of previous filter states and histories
//   -Reverting to previous states
//   -Processing inputs (controls or measurements) and handling out of order inputs
//   -Maintaining a queue of inputs which can be processed as a batch
//
// This does not provide any functionality for actually defining the prediction and correction
// steps, that is left to be implemented by the derived classes.
//
// The system and measurement models are independent of this class, they are provided as inputs.
// Thus, the filtering algorithm and the application are separate, and different applications can
// be used with the same algorithm.
//
template <typename SysT, typename MeasT>
class FilterBase {
  protected:
    // FilterInput
    //
    // Defines an input to the filter, which can be a control or measurement
    struct FilterInput {
        // Vector of control/measurement fields
        Eigen::VectorXd data;
        // Covariance matrix corresponding to data, only required for measurements
        Eigen::MatrixXd covariance;
        // Timestamp of the control/measurement [s]
        double timestamp = 0.0;
        // Flag for if this input is a control, or a measurement
        bool is_control = true;
        // Measurement model to use for processing a measurement, only required for measurements
        MeasT* model = nullptr;

        FilterInput() = default;

        FilterInput(const Eigen::VectorXd& data_in, const Eigen::MatrixXd& covariance_in,
                    double timestamp_in, bool is_control_in, MeasT* model_in = nullptr)
            : data(data_in)
            , covariance(covariance_in)
            , timestamp(timestamp_in)
            , is_control(is_control_in)
            , model(model_in) {}
    };

    // FilterState
    //
    // Represents the full state of the filter at any point
    struct FilterState {
        // Current filter state
        Eigen::VectorXd x;
        // Current filter covariance
        Eigen::MatrixXd covariance;
        // Timestamp corresponding to state
        double timestamp = 0.0;
        // The last input that was applied (can be a control or a measurement)
        FilterInput prev_input;
    };

  public:
    // All parameters for the filter
    struct Parameters {
        // When true a history of filter states and inputs will be kept, and when an old control
        // or measurment is received in the predict or correct methods the filter will rewind to
        // apply the input then re apply any previous inputs.
        bool rewind_history = false;
        // Duration to retain a history of the filter states [s]
        double history_window_sec = 1.0;
        // When true all state vector elements that contain angles will be constrained to be
        // within the interval [-pi, pi] after every predict and correction step
        bool constrain_angles = false;
        // State vector indices that contain angles
        std::vector<uint32_t> angle_indices;
    };

    // Constructor with state initialized to empty vectors and matrices
    //
    // @param system_model: Model describing the system dynamics
    FilterBase(SysT* system_model);

    // Constructor that initializes the filter's state
    //
    // @param system_model: Model describing the system dynamics
    // @param x: Initial filter state
    // @param cov: Intiial state covariance
    // @param timestamp: Initial state timestamp
    FilterBase(SysT* system_model, const Eigen::VectorXd& x, const Eigen::MatrixXd& cov,
               double timestamp);

    // parameters
    //
    // @return: Current parameter set for the filter
    Parameters& parameters();
    const Parameters& parameters() const;

    // initialize
    //
    // Initializes the filter to the provided state.
    //
    // This will clear any existing queues of controls or measurements, as well as the history of
    // the internal filter state for processing out of order data inputs.
    //
    // @param x: Initial filter state
    // @param cov: Intiial state covariance
    // @param timestamp: Initial state timestamp
    void initialize(const Eigen::VectorXd& x, const Eigen::MatrixXd& cov, double timestamp);

    // getState
    //
    // @return: Current state estimate
    Eigen::VectorXd getState() const;

    // getCovariance
    //
    // @return: Covariance of the current state estimate
    Eigen::MatrixXd getCovariance() const;

    // getStateTime
    //
    // @return: Timestamp of the current state estimate
    double getStateTime() const;

    // predict
    //
    // Applies a single prediction step to the filter my propegating the current state forward
    // (no control input).
    //
    // If the provided time is in the past, it will be ignored. This will not rewind the state.
    //
    // This will not change the existing queue of controls.
    //
    // @param Timestamp: Time to propegate the state forward to [s]
    void predict(double timestamp);

    // predict
    //
    // Applies a single prediction step to the filter with the provided controls.
    //
    // This will not change the existing queue of controls.
    //
    // @param u: Control input vector
    // @param timestamp: Time of the control input [s]
    void predict(const Eigen::VectorXd& u, double timestamp);

    // correct
    //
    // Applies a single correction step to the filter with the provided measurement.
    //
    // This will not change the existing queue of measurements.
    //
    // @param z: Measurement vector
    // @param covariance: Covariance matrix of the measurement
    // @param timestamp: Time of the control input [s]
    // @param model: Measurement model to use for processing the measurement
    void correct(const Eigen::VectorXd& z, const Eigen::MatrixXd& covariance, double timestamp,
                 MeasT* model);

    // enqueuControl
    //
    // Adds a control to the queue to be processed. will not process and controls in the queue.
    //
    // @param u: Control input vector
    // @param timestamp: Time of the control input [s]
    void enqueuControl(const Eigen::VectorXd& u, double timestamp);

    // enqueuMeasurement
    //
    // Adds a measurement to the queue to be processed. will not process and measurements in the
    // queue.
    //
    // @param z: Measurement vector
    // @param covariance: Covariance matrix of the measurement
    // @param timestamp: Time of the control input [s]
    // @param model: Measurement model to use for processing the measurement
    void enqueuMeasurement(const Eigen::VectorXd& z, const Eigen::MatrixXd& covariance,
                           double timestamp, MeasT* model);

    // processQueues
    //
    // Processes all controls and measurements present in the queues in temporal order.
    void processQueues();

    // clearControlQueue
    //
    // Clears all controls stored in the control queue.
    void clearControlQueue();

    // clearMeasurementQueue
    //
    // Clears all measurements stored in the measurement queue.
    void clearMeasurementQueue();

    // revertToState
    //
    // Attempts to revert to the closest state before the provided timestamp.
    //
    // Note, this will clear the control and measurement queues.
    //
    // @param timestamp: Timestamp to revert to (filter state will be before this stamp)
    // @return: True when there was a previous state to revert to or all states are before
    //          timestamp, false otherwise
    bool revertToState(double timestamp);

  protected:
    // clear
    //
    // Clears the filter state history and queues.
    void clear();

    // rewindHistory
    //
    // Attempts to revert to the closest state before the provided timestamp, and extracts all
    // states in
    // the history that occurred on or after the timestamp.
    //
    // @param timestamp: Timestamp to revert to (filter state will be before this stamp)
    // @param post_states: Container to store filter states that occurred after timestamp
    // @return: True when there was a previous state to revert to or all states are before
    // timestamp, false otherwise
    bool rewindHistory(double timestamp, std::vector<FilterState>* post_states);

    // Internal implementation of predict() for derived classes to populate.
    // Note, the time delta is provided instead of the absolute time.
    virtual void myPredict(double dt) = 0;
    virtual void myPredict(const Eigen::VectorXd& u, double dt) = 0;

    // Internal implementation of correct() for derived classes to populate
    virtual void myCorrect(const Eigen::VectorXd& z, MeasT* model) = 0;

    // Parameters for this object
    Parameters params_;
    // Current state of the filter
    FilterState filter_state_;
    // Model describing the system dynamics
    SysT* system_model_;

  private:
    // applyInput
    //
    // Applies a single input to the filters, which can be a control or measurement. This
    // essentially just calls the prediction or correction methods.
    //
    // @param input: Filter input to apply
    void applyInput(const FilterInput& input);

    // applyBatchInputs
    //
    // Applies a batch of inputs to the filters, which can be controls or measurements. This
    // essentially just calls the prediction and correction methods.
    //
    // @param inputs: Time ordered inputs to apply
    void applyBatchInputs(const std::vector<FilterInput>& inputs);

    // pruneHistory
    //
    // Removes all entries in the filter state history older than the window size.
    void pruneHistory();

    // pruneQueue
    //
    // Removes all entries in a queue older than the oldest filter state in the history (i.e.
    // inputs)
    // that can no longer be applied
    //
    // @param queue: Queue to prune
    void pruneQueue(std::deque<FilterInput>* queue);

    // constrainStateAngles
    //
    // Constrains all state vector angles to the interval [-pi, pi].
    void constrainStateAngles();

    // History of filter state for jumping back in time
    std::deque<FilterState> history_;
    // Queue of controls to process in the filter
    std::deque<FilterInput> control_queue_;
    // Queue of measurements to process in the filter
    std::deque<FilterInput> measurement_queue_;
};

}  // namespace state_estimation

#include "filter_base.hpp"
