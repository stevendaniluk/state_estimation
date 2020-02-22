#include <gtest/gtest.h>
#include <state_estimation/filters/kalman_filter.h>
#include "filter_test_utilities.h"

using namespace state_estimation;

// Since the KalmFilter class is only responsible for implementing the EKF update equations, only
// the correctness of those equations are tested

// Create a system model to use for tests. It simply adds the control to the state with:
//   x' = Ax + dt * u
class KFSampleSystemModel : public system_models::LinearSystemModel {
  public:
    KFSampleSystemModel(uint32_t n, uint32_t m)
        : LinearSystemModel::LinearSystemModel(n, m) {
        setA(Eigen::MatrixXd::Identity(n, n));
        R_p_ = 1e-3 * Eigen::MatrixXd::Identity(n, n);
        P_ = Eigen::MatrixXd::Identity(n, n);
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) override {
        setB(dt * Eigen::MatrixXd::Identity(x.size(), x.size()));
    }
};

// Create a measurement model to use for tests that that simply sets the predicted state as the
// measurement vector. The C matrix will be an identity matrix. This will make it easy to
// introspect the update equations.
class KFSampleMeasurementModel : public measurement_models::LinearMeasurementModel {
  public:
    KFSampleMeasurementModel(uint32_t n, uint32_t k)
        : LinearMeasurementModel::LinearMeasurementModel(n, k, false) {
        setC(Eigen::MatrixXd::Identity(k, n));
    }

  protected:
    void myUpdate(const Eigen::VectorXd& x, double dt) override {}
};

// Pass in our filter, system and measurement types to the filter test fixture
class KalmanFilterTest
    : public FilterTest<KalmanFilter, KFSampleSystemModel, KFSampleMeasurementModel> {};

TEST_F(KalmanFilterTest, PredictUsesAxPlusBuForStateUpdate) {
    double dt = 0.4;

    // Modify our A and B matrices
    Eigen::MatrixXd A(2, 2);
    A << 1, 2, 3, 4;
    Eigen::MatrixXd B(2, 2);
    B << 5, 6, 7, 8;

    // Run it through the filter with the updated matrices
    system_model.setA(A);
    system_model.setB(B);
    filter->predict(vec_22, filter->getStateTime() + dt);

    // Our new state should be: x' = Ax + Bu
    KFSampleSystemModel ref_model(2, 2);
    ref_model.update(x_i, vec_22, dt);
    ref_model.setA(A);
    Eigen::VectorXd x_target = ref_model.A() * x_i + ref_model.B() * vec_22;

    EXPECT_TRUE(filter->getState().isApprox(x_target, 1e-6))
        << "Target: " << x_target.transpose() << ", Actual: " << filter->getState().transpose();
}

TEST_F(KalmanFilterTest, PredictUsesAMatrixForCovarianceUpdate) {
    double dt = 0.4;

    // Modify our A matrix
    Eigen::MatrixXd A(2, 2);
    A << 1, 2, 3, 4;

    // Run it through the filter, with zero process noise to isolate the Jacobian
    system_model.setA(A);
    system_model.setProcessCovariance(Eigen::MatrixXd::Zero(2, 2));
    filter->predict(vec_22, filter->getStateTime() + dt);

    // Compute the target covariance with the Kalman update of Sigma = A * Sigma * A'
    Eigen::MatrixXd cov_target = A * cov_i * A.transpose();

    EXPECT_TRUE(filter->getCovariance().isApprox(cov_target, 1e-6)) << "Target:" << std::endl
                                                                    << cov_target << std::endl
                                                                    << "Actual:" << std::endl
                                                                    << filter->getCovariance();
}

TEST_F(KalmanFilterTest, PredictUsesSystemModelProcessNoise) {
    predictUsesSystemModelProcessNoise();
}

TEST_F(KalmanFilterTest, ZeroMeasurementCovarianceProducesZeroEstimateCovariance) {
    zeroMeasurementCovarianceProducesZeroEstimateCovariance();
}

TEST_F(KalmanFilterTest, ZeroMeasurementCovarianceSetsStateToPredictedMeasurement) {
    zeroMeasurementCovarianceSetsStateToPredictedMeasurement();
}

TEST_F(KalmanFilterTest, VeryLargeMeasurementCovarianceMakesMeasurementIgnored) {
    veryLargeMeasurementCovarianceMakesMeasurementIgnored();
}

TEST_F(KalmanFilterTest, CorrectWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement) {
    correctWithEqualCovarianceUpdatesToMeanOfStateAndMeasurement();
}

TEST_F(KalmanFilterTest, CorrectWithEqualCovarianceHalvesTheCovarience) {
    correctWithEqualCovarianceHalvesTheCovarience();
}
