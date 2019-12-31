#include <gtest/gtest.h>
#include <state_estimation/filters/filter_base.h>
#include <state_estimation/measurement_models/measurement_model.h>
#include <state_estimation/system_models/system_model.h>

using namespace state_estimation;

// Simple system model for the test filter defined below. There is nothing special about this, it
// simply has to perform some arithmetic on the input state and control so we can compare
// predictions.
class TestFilterSystemModel : public SystemModel {
  public:
    TestFilterSystemModel()
        : SystemModel::SystemModel(3, 3) {}

    Eigen::VectorXd prediction() { return x_pred_; }

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override {
        x_pred_ = x + dt * x + u;
    }

    void myUpdateNoControl(const Eigen::VectorXd& x, double dt) override { x_pred_ = x + dt * x; }

    Eigen::VectorXd x_pred_;
};

// Simple measurement model for the test filter defined below. There is nothing special about this,
// it simply has to perform some arithmetic on the input state and control so we can compare
// predictions.
class TestFilterMeasurementModel : public MeasurementModel {
  public:
    TestFilterMeasurementModel()
        : MeasurementModel::MeasurementModel(3, 3) {}

    Eigen::VectorXd prediction() { return z_pred_; }

  protected:
    void myUpdate(const Eigen::VectorXd& x) override { z_pred_ = 0.9 * x; }

    Eigen::VectorXd z_pred_;
};

// Filter implementation for testing purposes.
//
// This uses the models defined above. During the correction step the state is set directly from
// the predicted measurement. This makes it easier to evaluate the correction method because we
// can compare it to another instance of the measurement model.
//
// This filter does nothing with the covariance values.
class TestFilter : public FilterBase<TestFilterSystemModel, TestFilterMeasurementModel> {
  public:
    using FilterBase<TestFilterSystemModel, TestFilterMeasurementModel>::FilterBase;

  protected:
    void myPredict(double dt) override {
        system_model_->updateNoControl(getState(), dt);
        filter_state_.x = system_model_->prediction();
    }

    void myPredict(const Eigen::VectorXd& control, double dt) override {
        system_model_->update(getState(), control, dt);
        filter_state_.x = system_model_->prediction();
    }

    void myCorrect(const Eigen::VectorXd& measurement, TestFilterMeasurementModel* model) override {
        model->update(getState());
        filter_state_.x = model->prediction();
    }
};

// Test fixture for storing a filter and an assortment of input data
class FilterBaseTest : public ::testing::Test {
  protected:
    void SetUp() override { filter.reset(new TestFilter(&system_model, x_i, cov_i, t_i)); }

    std::unique_ptr<TestFilter> filter;
    TestFilterSystemModel system_model;
    TestFilterMeasurementModel meas_model;

    // Assortment of input vectors
    Eigen::VectorXd vec_000 = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd vec_111 = Eigen::VectorXd::Constant(3, 1);
    Eigen::VectorXd vec_222 = Eigen::VectorXd::Constant(3, 2);
    Eigen::VectorXd vec_n111 = Eigen::VectorXd::Constant(3, -1);
    Eigen::VectorXd vec_n222 = Eigen::VectorXd::Constant(3, -2);

    // Initial conditions for the filter
    Eigen::VectorXd x_i = vec_111;
    Eigen::MatrixXd cov_i = Eigen::MatrixXd::Identity(3, 3);
    double t_i = 3.14159;
};

TEST_F(FilterBaseTest, InitializeSetsStateCoverianceAndTime) {
    double t = 123.45;
    Eigen::MatrixXd cov(2, 2);
    cov << 1.1, 2.2, 3.3, 4.4;

    filter->initialize(vec_222, cov, t);

    EXPECT_EQ(vec_222, filter->getState())
        << "Target: " << vec_222.transpose() << ", Actual: " << filter->getState().transpose();
    EXPECT_EQ(cov, filter->getCovariance()) << "Target:" << std::endl
                                            << cov << ", Actual:" << std::endl
                                            << filter->getCovariance();
    EXPECT_EQ(t, filter->getStateTime());
}

TEST_F(FilterBaseTest, CorrectAdvancesStateBeforeCorrecting) {
    // Manually calculate what the state should get predicted to be using the system and
    // measurement models. Our test filter sets the corrected state to be equal to the predicted
    // measurement.
    double dt = 0.2;
    Eigen::VectorXd z = vec_222;

    TestFilterSystemModel ref_system_model;
    ref_system_model.updateNoControl(x_i, dt);
    Eigen::VectorXd x_pred = ref_system_model.prediction();

    TestFilterMeasurementModel ref_meas_model;
    ref_meas_model.update(x_pred);
    Eigen::VectorXd x_target = ref_meas_model.prediction();

    filter->correct(z, cov_i, t_i + dt, &meas_model);

    EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, MeasurementQueueAdvancesStateBeforeCorrecting) {
    // Manually calculate what the state should get predicted to be using the system and
    // measurement models. Our test filter sets the corrected state to be equal to the predicted
    // measurement.
    double dt = 0.2;
    Eigen::VectorXd z = vec_222;

    TestFilterSystemModel ref_system_model;
    ref_system_model.updateNoControl(x_i, dt);
    Eigen::VectorXd x_pred = ref_system_model.prediction();

    TestFilterMeasurementModel ref_meas_model;
    ref_meas_model.update(x_pred);
    Eigen::VectorXd x_target = ref_meas_model.prediction();

    filter->enqueuMeasurement(z, cov_i, t_i + dt, &meas_model);
    filter->processQueues();

    EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, ControlQueueAppliedInOrder) {
    // Make two filters, one will get fed controls individually in order, the other will get them
    // added to the queue out of order, they should result in the same state.
    TestFilter filter_ref(&system_model, x_i, cov_i, t_i);
    TestFilter filter_queue(&system_model, x_i, cov_i, t_i);

    std::vector<std::pair<Eigen::VectorXd, double>> inputs;
    inputs.emplace_back(vec_111, t_i + 0.1);
    inputs.emplace_back(vec_222, t_i + 0.3);
    inputs.emplace_back(vec_111, t_i + 0.9);

    for (auto input : inputs) {
        filter_ref.predict(input.first, input.second);
    }

    for (int i : {1, 0, 2}) {
        filter_queue.enqueuControl(inputs[i].first, inputs[i].second);
    }
    filter_queue.processQueues();

    EXPECT_TRUE(filter_queue.getState().isApprox(filter_ref.getState(), 1e-6))
        << "Target: " << filter_ref.getState().transpose()
        << ", Actual: " << filter_queue.getState().transpose();
}

TEST_F(FilterBaseTest, MeasurementQueueAppliedInOrder) {
    // Make two filters, one will get fed measurements individually in order, the other will get
    // them added to the queue out of order, they should result in the same state.
    TestFilter filter_ref(&system_model, x_i, cov_i, t_i);
    TestFilter filter_queue(&system_model, x_i, cov_i, t_i);

    std::vector<std::pair<Eigen::VectorXd, double>> inputs;
    inputs.emplace_back(vec_111, t_i + 0.1);
    inputs.emplace_back(vec_222, t_i + 0.3);
    inputs.emplace_back(vec_111, t_i + 0.9);

    for (auto input : inputs) {
        filter_ref.correct(input.first, cov_i, input.second, &meas_model);
    }

    for (int i : {1, 0, 2}) {
        filter_queue.enqueuMeasurement(inputs[i].first, cov_i, inputs[i].second, &meas_model);
    }
    filter_queue.processQueues();

    EXPECT_TRUE(filter_queue.getState().isApprox(filter_ref.getState(), 1e-6))
        << "Target: " << filter_ref.getState().transpose()
        << ", Actual: " << filter_queue.getState().transpose();
}

TEST_F(FilterBaseTest, BothQueuesAppliedInOrder) {
    // Make two filters, provide all inputs to one in order, and add them to the queue of the other
    // filter out of order, both should result in the same state.
    TestFilter filter_ref(&system_model, x_i, cov_i, t_i);
    TestFilter filter_queue(&system_model, x_i, cov_i, t_i);

    std::vector<std::pair<Eigen::VectorXd, double>> inputs;
    inputs.emplace_back(vec_111, t_i + 0.1);
    inputs.emplace_back(vec_222, t_i + 0.3);
    inputs.emplace_back(vec_111, t_i + 0.9);

    filter_ref.predict(inputs[0].first, inputs[0].second);
    filter_ref.correct(inputs[1].first, cov_i, inputs[1].second, &meas_model);
    filter_ref.predict(inputs[2].first, inputs[2].second);

    filter_queue.enqueuMeasurement(inputs[1].first, cov_i, inputs[1].second, &meas_model);
    filter_queue.enqueuControl(inputs[2].first, inputs[2].second);
    filter_queue.enqueuControl(inputs[0].first, inputs[0].second);
    filter_queue.processQueues();

    EXPECT_TRUE(filter_queue.getState().isApprox(filter_ref.getState(), 1e-6))
        << "Target: " << filter_ref.getState().transpose()
        << ", Actual: " << filter_queue.getState().transpose();
}

TEST_F(FilterBaseTest, PredictWithOldControlWillReApplyInputs) {
    // Create two filters, one will get controls applied in order, the other will be applied out of
    // order, they both should have the same state
    TestFilter filter_ref(&system_model, x_i, cov_i, t_i);
    TestFilter filter_old_control(&system_model, x_i, cov_i, t_i);

    filter_ref.parameters().rewind_history = true;
    filter_old_control.parameters().rewind_history = true;

    std::vector<std::pair<Eigen::VectorXd, double>> inputs;
    inputs.emplace_back(vec_111, t_i + 0.1);
    inputs.emplace_back(vec_222, t_i + 0.3);

    filter_ref.predict(inputs[0].first, inputs[0].second);
    filter_ref.predict(inputs[1].first, inputs[1].second);

    filter_old_control.predict(inputs[1].first, inputs[1].second);
    filter_old_control.predict(inputs[0].first, inputs[0].second);

    EXPECT_TRUE(filter_old_control.getState().isApprox(filter_ref.getState(), 1e-6))
        << "Target: " << filter_ref.getState().transpose()
        << ", Actual: " << filter_old_control.getState().transpose();
}

TEST_F(FilterBaseTest, CorrectWithOldMeasurementWillReApplyInputs) {
    // Create two filters, one will get measurements applied in order, the other will be applied out
    // of order, they both should have the same state
    TestFilter filter_ref(&system_model, x_i, cov_i, t_i);
    TestFilter filter_old_meas(&system_model, x_i, cov_i, t_i);

    filter_ref.parameters().rewind_history = true;
    filter_old_meas.parameters().rewind_history = true;

    std::vector<std::pair<Eigen::VectorXd, double>> inputs;
    inputs.emplace_back(vec_111, t_i + 0.1);
    inputs.emplace_back(vec_222, t_i + 0.3);

    filter_ref.correct(inputs[0].first, cov_i, inputs[0].second, &meas_model);
    filter_ref.correct(inputs[1].first, cov_i, inputs[1].second, &meas_model);

    filter_old_meas.correct(inputs[1].first, cov_i, inputs[1].second, &meas_model);
    filter_old_meas.correct(inputs[0].first, cov_i, inputs[0].second, &meas_model);

    EXPECT_TRUE(filter_old_meas.getState().isApprox(filter_ref.getState(), 1e-6))
        << "Target: " << filter_ref.getState().transpose()
        << ", Actual: " << filter_old_meas.getState().transpose();
}

TEST_F(FilterBaseTest, PredictNoControlIgnoresTimestampNotInTheFuture) {
    // The filter has a non zero state so our mock system model will change it for a non zero
    // time advancement. It also does not have any prior history before so it cannot rewind and re
    // apply inputs
    filter->predict(t_i);
    EXPECT_TRUE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();

    filter->predict(t_i - 0.001);
    EXPECT_TRUE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, PredictIgnoresTimestampNotInTheFuture) {
    // The filter has a non zero state so our mock system model will change it for a non zero
    // time advancement. It also does not have any prior history before so it cannot rewind and re
    // apply inputs
    filter->predict(vec_222, t_i);
    EXPECT_TRUE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();

    filter->predict(vec_222, t_i - 0.001);
    EXPECT_TRUE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, MeasurementInPastIsIgnored) {
    // The filter does not have any prior history before so it cannot rewind and re apply inputs
    filter->correct(vec_222, cov_i, t_i - 0.001, &meas_model);
    EXPECT_TRUE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, MeasurementWithSameTimestampIsAccepted) {
    filter->correct(vec_222, cov_i, t_i, &meas_model);
    EXPECT_FALSE(filter->getState().isApprox(x_i, 1e-6))
        << "Target: " << x_i.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, RevertToStateUsesStateBeforeTimestamp) {
    filter->parameters().rewind_history = true;

    // Step the filter forward with some inputs
    double t_2 = t_i + 1.0;
    filter->predict(vec_111, t_2);
    Eigen::VectorXd t_2_state = filter->getState();

    double t_3 = t_2 + 1.0;
    filter->predict(vec_222, t_3);
    Eigen::VectorXd t_3_state = filter->getState();

    double t_4 = t_3 + 1.0;
    filter->predict(vec_111, t_4);

    double t_5 = t_4 + 1.0;
    filter->predict(vec_222, t_5);

    // Revert exactly to the t_4 time, should end up with the t_3 state
    ASSERT_TRUE(filter->revertToState(t_4));
    ASSERT_EQ(t_3_state, filter->getState())
        << "Target: " << t_3_state.transpose() << ", Actual: " << filter->getState().transpose();

    // Revert to right afte the t_2 state, should end up with the t_2 state
    ASSERT_TRUE(filter->revertToState(t_2 + 0.001));
    ASSERT_EQ(t_2_state, filter->getState())
        << "Target: " << t_2_state.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, RevertToStateFailsWithTooOldTimestamp) {
    // Provide some inputs, just so there is a history to evaluate
    filter->predict(vec_111, t_i + 1.0);
    filter->predict(vec_222, t_i + 2.0);
    filter->predict(vec_000, t_i + 3.0);

    Eigen::VectorXd pre_revert_state = filter->getState();
    ASSERT_FALSE(filter->revertToState(t_i - 0.001));
    EXPECT_EQ(pre_revert_state, filter->getState())
        << "Target: " << pre_revert_state.transpose()
        << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, RevertToStateDoesNothingWithFutureDatedTimestamp) {
    // Provide some inputs, just so there is a history to evaluate
    filter->predict(vec_111, t_i + 1.0);
    filter->predict(vec_222, t_i + 2.0);
    filter->predict(vec_000, t_i + 3.0);

    Eigen::VectorXd pre_revert_state = filter->getState();
    ASSERT_TRUE(filter->revertToState(t_i + 10.0));
    EXPECT_EQ(pre_revert_state, filter->getState())
        << "Target: " << pre_revert_state.transpose()
        << ", Actual: " << filter->getState().transpose();
}

TEST_F(FilterBaseTest, PredictSkippedWhenStationary) {
    // Define some custom functions for the system model stationary ops that declare the system
    // stationary for positive values in the first entries of the control vector, and simply
    // increments the state and covariance by some factor
    std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)> isStationary =
        [](const Eigen::VectorXd& x, const Eigen::VectorXd& data) { return data(0) > 0; };

    std::function<void(Eigen::VectorXd*, Eigen::MatrixXd*)> makeStationary =
        [](Eigen::VectorXd* x, Eigen::MatrixXd* cov) {
            (*x) += Eigen::VectorXd::Constant(x->size(), 1000);
            (*cov) += Eigen::MatrixXd::Constant(cov->rows(), cov->cols(), 1000);
        };

    system_model.setIsStationaryFunction(isStationary);
    system_model.setMakeStationaryFunction(makeStationary);
    system_model.setCheckStationary(true);

    Eigen::VectorXd x_stationary;
    Eigen::MatrixXd cov_stationary;

    // With a zero control input the state should not be stationary
    x_stationary = filter->getState();
    cov_stationary = filter->getCovariance();
    makeStationary(&x_stationary, &cov_stationary);

    filter->predict(vec_000, filter->getStateTime() + 1.0);
    EXPECT_NE(x_stationary, filter->getState())
        << "Target: " << x_stationary.transpose() << ", Actual: " << filter->getState().transpose();
    EXPECT_NE(cov_stationary, filter->getCovariance()) << "Target:" << std::endl
                                                       << cov_stationary << std::endl
                                                       << "Actual:" << std::endl
                                                       << filter->getCovariance();

    // Non zero control should be stationary
    x_stationary = filter->getState();
    cov_stationary = filter->getCovariance();
    makeStationary(&x_stationary, &cov_stationary);

    filter->predict(vec_111, filter->getStateTime() + 1.0);
    EXPECT_EQ(x_stationary, filter->getState())
        << "Target: " << x_stationary.transpose() << ", Actual: " << filter->getState().transpose();
    EXPECT_EQ(cov_stationary, filter->getCovariance()) << "Target:" << std::endl
                                                       << cov_stationary << std::endl
                                                       << "Actual:" << std::endl
                                                       << filter->getCovariance();
}

TEST_F(FilterBaseTest, CorrectSkippedWhenStationary) {
    // Define some custom functions for the measurement model stationary ops that declare the
    // system stationary for positive values in the first entries of the measurement vector, and
    // simply increments the state and covariance by some factor
    std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)> isStationary =
        [](const Eigen::VectorXd& x, const Eigen::VectorXd& data) { return data(0) > 0; };

    std::function<void(Eigen::VectorXd*, Eigen::MatrixXd*)> makeStationary =
        [](Eigen::VectorXd* x, Eigen::MatrixXd* cov) {
            (*x) += Eigen::VectorXd::Constant(x->size(), 1000);
            (*cov) += Eigen::MatrixXd::Constant(cov->rows(), cov->cols(), 1000);
        };

    meas_model.setIsStationaryFunction(isStationary);
    meas_model.setMakeStationaryFunction(makeStationary);
    meas_model.setCheckStationary(true);

    Eigen::VectorXd x_stationary;
    Eigen::MatrixXd cov_stationary;

    // With a zero measurement input the state should not be stationary
    x_stationary = filter->getState();
    cov_stationary = filter->getCovariance();
    makeStationary(&x_stationary, &cov_stationary);

    filter->correct(vec_000, cov_i, filter->getStateTime(), &meas_model);
    EXPECT_NE(x_stationary, filter->getState())
        << "Target: " << x_stationary.transpose() << ", Actual: " << filter->getState().transpose();
    EXPECT_NE(cov_stationary, filter->getCovariance()) << "Target:" << std::endl
                                                       << cov_stationary << std::endl
                                                       << "Actual:" << std::endl
                                                       << filter->getCovariance();

    // Non zero measurement should be stationary
    x_stationary = filter->getState();
    cov_stationary = filter->getCovariance();
    makeStationary(&x_stationary, &cov_stationary);

    filter->correct(vec_111, cov_i, filter->getStateTime(), &meas_model);
    EXPECT_EQ(x_stationary, filter->getState())
        << "Target: " << x_stationary.transpose() << ", Actual: " << filter->getState().transpose();
    EXPECT_EQ(cov_stationary, filter->getCovariance()) << "Target:" << std::endl
                                                       << cov_stationary << std::endl
                                                       << "Actual:" << std::endl
                                                       << filter->getCovariance();
}
