namespace state_estimation {

template <typename SysT, typename MeasT>
FilterBase<SysT, MeasT>::FilterBase(SysT* system_model)
    : FilterBase<SysT, MeasT>::FilterBase(system_model, Eigen::VectorXd(), Eigen::MatrixXd(), 0.0) {
}

template <typename SysT, typename MeasT>
FilterBase<SysT, MeasT>::FilterBase(SysT* system_model, const Eigen::VectorXd& x,
                                    const Eigen::MatrixXd& cov, double timestamp)
    : system_model_(system_model) {
    initialize(x, cov, timestamp);
}

template <typename SysT, typename MeasT>
typename FilterBase<SysT, MeasT>::Parameters& FilterBase<SysT, MeasT>::parameters() {
    return params_;
}

template <typename SysT, typename MeasT>
const typename FilterBase<SysT, MeasT>::Parameters& FilterBase<SysT, MeasT>::parameters() const {
    return params_;
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::initialize(const Eigen::VectorXd& x, const Eigen::MatrixXd& cov,
                                         double timestamp) {
    clear();

    filter_state_.x = x;
    filter_state_.covariance = cov;
    filter_state_.timestamp = timestamp;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "Filter state initialized to: t=" << std::to_string(filter_state_.timestamp)
              << "s, x=" << printMatrix(filter_state_.x) << ", covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif
}

template <typename SysT, typename MeasT>
Eigen::VectorXd FilterBase<SysT, MeasT>::getState() const {
    return filter_state_.x;
}

template <typename SysT, typename MeasT>
Eigen::MatrixXd FilterBase<SysT, MeasT>::getCovariance() const {
    return filter_state_.covariance;
}

template <typename SysT, typename MeasT>
double FilterBase<SysT, MeasT>::getStateTime() const {
    return filter_state_.timestamp;
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::predict(double timestamp) {
    if (timestamp > filter_state_.timestamp) {
        const FilterInput input(Eigen::VectorXd(), Eigen::MatrixXd(), timestamp, true);
        applyInput(input);
        pruneHistory();
    }
#ifdef DEBUG_STATE_ESTIMATION
    else {
        std::cout << "Ignoring prediction because timestamp is in the past" << std::endl;
        return;
    }
#endif
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::predict(const Eigen::VectorXd& u, double timestamp) {
    if (!params_.rewind_history) {
        if (timestamp > filter_state_.timestamp) {
            const FilterInput input(u, Eigen::MatrixXd(), timestamp, true);
            applyInput(input);
            pruneHistory();
        }
#ifdef DEBUG_STATE_ESTIMATION
        else {
            std::cout << "Ignoring prediction because timestamp is in the past" << std::endl;
            return;
        }
#endif
    } else {
        // In the event this control is older than the current state estimate, we'll need to rewind
        // the state, apply this control, then re apply all the existing inputs. So we'll form a
        // list of inputs to apply, with this control being first.
        std::vector<FilterInput> inputs;

        // The covariance for controls will be determined by the system model at the time the
        // control is processed since it may be a function of the state, so we'll provide a dummy
        // matrix
        inputs.emplace_back(u, Eigen::MatrixXd(), timestamp, true);

        std::vector<FilterState> prev_states;
        if (timestamp < filter_state_.timestamp && !rewindHistory(timestamp, &prev_states)) {
            // There is no state in the history prior to this control, so we cannot process it
            std::cerr << "Cannot apply control input with timestamp " << std::to_string(timestamp)
                      << "s because it is older than any states in the filter's history"
                      << std::endl;
            return;
        }
        for (const auto& state : prev_states) {
            inputs.push_back(state.prev_input);
        }

        applyBatchInputs(inputs);
    }
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::correct(const Eigen::VectorXd& z, const Eigen::MatrixXd& covariance,
                                      double timestamp, MeasT* model) {
    if (!params_.rewind_history) {
        const double dt = timestamp - filter_state_.timestamp;
        if (dt >= 0) {
            const FilterInput input(z, covariance, timestamp, false, model);
            applyInput(input);
            pruneHistory();
        }
#ifdef DEBUG_STATE_ESTIMATION
        else {
            std::cout << "Ignoring correction because timestamp is in the past" << std::endl;
            return;
        }
#endif
    } else {
        // In the event this measurement is older than the current state estimate, we'll need to
        // rewind the state, apply this measurement, then re apply all the existing inputs. So we'll
        // form a list of inputs to apply, with this measurement being first.
        std::vector<FilterInput> inputs;
        inputs.emplace_back(z, covariance, timestamp, false, model);

        std::vector<FilterState> prev_states;
        if (timestamp < filter_state_.timestamp && !rewindHistory(timestamp, &prev_states)) {
            // There is no state in the history prior to this measurement, so we cannot process it
            std::cerr << "Cannot apply measurement input with timestamp "
                      << std::to_string(timestamp)
                      << "s because it is older than any states in the filter's history"
                      << std::endl;
            return;
        }
        for (const auto& state : prev_states) {
            inputs.push_back(state.prev_input);
        }

        applyBatchInputs(inputs);
    }
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::enqueuControl(const Eigen::VectorXd& u, double timestamp) {
    // The covariance for controls will be determed by the system model at the time the control is
    // processed since it may be a function of the state, so we'll provide a dummy matrix
    control_queue_.emplace_back(u, Eigen::MatrixXd(), timestamp, true);
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::enqueuMeasurement(const Eigen::VectorXd& z,
                                                const Eigen::MatrixXd& covariance, double timestamp,
                                                MeasT* model) {
    measurement_queue_.emplace_back(z, covariance, timestamp, false, model);
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::processQueues() {
#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "Processing control and measurement queues" << std::endl;
#endif
    // All the inputs need to be processed in order, and we also need to account for the possibility
    // of data in the queue being older than the current estimate.
    //
    // So, we'll first rewind the state, then sort all the already applied inputs and the queue
    // inputs by time, then apply them all sequentially.

    // Have to prune out any inputs older than our history, then see what the oldest input is
    pruneQueue(&control_queue_);
    pruneQueue(&measurement_queue_);

    double oldest_timestamp = std::numeric_limits<double>::infinity();
    for (const auto& input : control_queue_) {
        oldest_timestamp = std::min(oldest_timestamp, input.timestamp);
    }
    for (const auto& input : measurement_queue_) {
        oldest_timestamp = std::min(oldest_timestamp, input.timestamp);
    }

    // Add all the inputs together and sort them
    std::vector<FilterInput> inputs;
    std::vector<FilterState> post_states;
    if (params_.rewind_history) {
        rewindHistory(oldest_timestamp, &post_states);
    }

    inputs.reserve(post_states.size() + control_queue_.size() + measurement_queue_.size());
    for (const auto& state : post_states) {
        inputs.push_back(state.prev_input);
    }
    inputs.insert(inputs.end(), control_queue_.begin(), control_queue_.end());
    inputs.insert(inputs.end(), measurement_queue_.begin(), measurement_queue_.end());

    std::sort(inputs.begin(), inputs.end(),
              [](const FilterInput& A, const FilterInput& B) { return A.timestamp < B.timestamp; });

    applyBatchInputs(inputs);
    clearControlQueue();
    clearMeasurementQueue();
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::clearControlQueue() {
    control_queue_.clear();
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::clearMeasurementQueue() {
    measurement_queue_.clear();
}

template <typename SysT, typename MeasT>
bool FilterBase<SysT, MeasT>::revertToState(double timestamp) {
#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "Reverting to state at tiemstamp " << std::to_string(timestamp) << "s"
              << std::endl;
#endif

    std::vector<FilterState> pose_states;
    if (rewindHistory(timestamp, &pose_states)) {
        // Need to clear all queues
        clearControlQueue();
        clearMeasurementQueue();

        return true;
    } else {
        return false;
    }
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::clear() {
    history_.clear();
    clearControlQueue();
    clearMeasurementQueue();
}

template <typename SysT, typename MeasT>
bool FilterBase<SysT, MeasT>::rewindHistory(double timestamp,
                                            std::vector<FilterState>* post_states) {
    if (timestamp > filter_state_.timestamp) {
#ifdef DEBUG_STATE_ESTIMATION
        std::cout << "Will not rewind history, requested time " << std::to_string(timestamp)
                  << "s is newer than estimate time " << std::to_string(filter_state_.timestamp)
                  << "s" << std::endl;
#endif
        // Nothing to do, all history is already older than the timestamp
        return true;
    } else if (!history_.empty() && timestamp > history_.front().timestamp) {
#ifdef DEBUG_STATE_ESTIMATION
        std::cout << "Rewinding history from " << std::to_string(filter_state_.timestamp)
                  << "s, to " << std::to_string(timestamp) << "s" << std::endl;

#endif
        // We'll record an internal list of states then copy them to the output, because we'll be
        // recording them in reverse order and will have to reverse the elements. We don't want to
        // modify any existing contents in the provided vector.
        std::vector<FilterState> post_states_internal;

        post_states_internal.push_back(filter_state_);

        // Extract everything newer than the provided timestamp. We'll go through the history in
        // reverse
        // order, since that is most efficient with a deque to remove entries. But, this means we
        // have
        // to reverse the order of the set of states we extracted.
        typename std::deque<FilterState>::reverse_iterator iter;
        for (iter = history_.rbegin(); iter != history_.rend(); ++iter) {
            if ((*iter).timestamp >= timestamp) {
                post_states_internal.push_back(*iter);
                history_.pop_back();
            }
        }

        std::reverse(post_states_internal.begin(), post_states_internal.end());
        post_states->insert(post_states->end(), post_states_internal.begin(),
                            post_states_internal.end());

        // The newest remaining entry in the history will be our current state
        filter_state_ = history_.back();
        history_.pop_back();

        return true;
    } else {
#ifdef DEBUG_STATE_ESTIMATION
        std::cout << "Cannot rewind history, requested time " << std::to_string(timestamp)
                  << "s is not present in the filter history" << std::endl;
#endif
        return false;
    }
}

template <typename SysT, typename MeasT>
std::string FilterBase<SysT, MeasT>::printMatrix(const Eigen::MatrixXd& mat) {
    std::ostringstream stream;
    stream << ((mat.cols() > 1) ? mat : mat.transpose())
                  .format(Eigen::IOFormat(6, 0, ", ", "\n", "[", "]"));
    return stream.str();
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::applyInput(const FilterInput& input) {
    if (params_.rewind_history) {
        history_.push_back(filter_state_);
    }

    const double dt = input.timestamp - filter_state_.timestamp;
    if (input.is_control) {
#ifdef DEBUG_STATE_ESTIMATION
        std::cout << "Applying prediction:" << std::endl
                  << "t=" << std::to_string(input.timestamp) << "s (dt=" << std::to_string(dt)
                  << ")" << std::endl
                  << "u=" << printMatrix(input.data) << std::endl
                  << "x=" << printMatrix(filter_state_.x) << std::endl
                  << "covariance=" << std::endl
                  << printMatrix(filter_state_.covariance) << std::endl;
#endif

        if (input.data.size() > 0) {
            myPredict(input.data, dt);
        } else {
            myPredict(dt);
        }
    } else {
        if (dt > 0) {
            // Need to first project the state forward to the measurement time

#ifdef DEBUG_STATE_ESTIMATION
            std::cout << "Advancing state x=" << printMatrix(filter_state_.x) << " by " << dt
                      << "s before applying correction" << std::endl;
#endif
            myPredict(dt);
        }

#ifdef DEBUG_STATE_ESTIMATION
        std::cout << "Applying measurement:" << std::endl
                  << "t=" << std::to_string(input.timestamp) << "s (dt=" << std::to_string(dt)
                  << ")" << std::endl
                  << "z=" << printMatrix(input.data) << std::endl
                  << "x=" << printMatrix(filter_state_.x) << std::endl
                  << "covariance=" << std::endl
                  << printMatrix(filter_state_.covariance) << std::endl;
#endif

        // We set the measurement model covariance here because it may be time varying
        input.model->setCovariance(input.covariance);
        myCorrect(input.data, input.model);
    }

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "Post update:" << std::endl
              << "x=" << printMatrix(filter_state_.x) << std::endl
              << "covariance=" << std::endl
              << printMatrix(filter_state_.covariance) << std::endl;
#endif

    filter_state_.prev_input = input;
    filter_state_.timestamp = input.timestamp;
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::applyBatchInputs(const std::vector<FilterInput>& inputs) {
    for (const auto& input : inputs) {
        applyInput(input);
    }
    pruneHistory();
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::pruneHistory() {
    if (params_.rewind_history) {
        return;
    }

    // Find the oldest allowed timestamp, and remove any entries older than that
    const double cutoff_timestamp = filter_state_.timestamp - params_.history_window_sec;

#ifdef DEBUG_STATE_ESTIMATION
    std::cout << "Pruning filter history older than " << std::to_string(cutoff_timestamp) << "s"
              << std::endl;
#endif

    for (auto iter = history_.rbegin(); iter != history_.rend(); ++iter) {
        if ((*iter).timestamp < cutoff_timestamp) {
            history_.pop_back();
        } else {
            break;
        }
    }
}

template <typename SysT, typename MeasT>
void FilterBase<SysT, MeasT>::pruneQueue(std::deque<FilterInput>* queue) {
    if (params_.rewind_history && !history_.empty()) {
        for (auto iter = queue->begin(); iter != queue->end();) {
            if (iter->timestamp < history_.front().timestamp) {
                iter = queue->erase(iter);
            } else {
                ++iter;
            }
        }
    }
}

}  // namespace state_estimation
