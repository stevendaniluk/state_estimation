#include <gtest/gtest.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>
#include <Eigen/Core>

namespace state_estimation {

// jacobianMatchesNumericalApproximation
//
// Compares the Jacobian produced by the system model to a numerical estimate formed by perturbing
// the state
//
// @param model: Model to evaluate
// @param x: State to evaluate the Jacobian about
// @param dt: Time step to use in the model update
// @param epsilon: Amount to perturb the state variables by
// @param tolerance: Tolerance on the Frobenius norm between the numerically determine Jacobian and
//                   the output of the system model
void jacobianMatchesNumericalApproximation(system_models::NonlinearSystemModel* model,
                                           const Eigen::VectorXd& x, double dt,
                                           double epsilon = 1e-6, double tolerance = 1e-3) {
    // Run our reference state through a model, then numerically compute the Jacobian by going
    // through each state variable perturbing it slightly.
    model->updateNoControl(x, dt);
    Eigen::VectorXd x_pred = model->g();
    Eigen::MatrixXd G_target = model->G();

    Eigen::MatrixXd G_num(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i) {
        // Perturb the state slightly
        Eigen::VectorXd x_pert = x;
        x_pert(i) += epsilon;

        // Run the new state through the model
        model->updateNoControl(x_pert, dt);

        // Compute the partial derivative
        G_num.col(i) = (model->g() - x_pred) / epsilon;
    }

    EXPECT_TRUE(G_num.isApprox(G_target, tolerance)) << "Target:" << std::endl
                                                     << G_target << std::endl
                                                     << "Actual:" << std::endl
                                                     << G_num;
}

// jacobianMatchesNumericalApproximation
//
// Compares the Jacobian produced by the measurement model to a numerical estimate formed by
// perturbing the state
//
// @param model: Model to evaluate
// @param x: State to evaluate the Jacobian about
// @param dt: Time delta to use
// @param epsilon: Amount to perturb the state variables by
// @param tolerance: Tolerance on the Frobenius norm between the numerically determine Jacobian and
//                   the output of the system model
void jacobianMatchesNumericalApproximation(measurement_models::NonlinearMeasurementModel* model,
                                           const Eigen::VectorXd& x, double dt,
                                           double epsilon = 1e-6, double tolerance = 1e-3) {
    // Run our reference state through a model, then numerically compute the Jacobian by going
    // through each state variable perturbing it slightly.
    model->update(x, dt);
    Eigen::VectorXd z_pred = model->h();
    Eigen::MatrixXd H_target = model->H();

    Eigen::MatrixXd H_num(H_target.rows(), H_target.cols());
    for (int i = 0; i < x.size(); ++i) {
        // Perturb the state slightly
        Eigen::VectorXd x_pert = x;
        x_pert(i) += epsilon;

        // Run the new state through the model
        model->update(x_pert, dt);

        // Compute the partial derivative
        H_num.col(i) = (model->h() - z_pred) / epsilon;
    }

    EXPECT_TRUE(H_num.isApprox(H_target, tolerance)) << "Target:" << std::endl
                                                     << H_target << std::endl
                                                     << "Actual:" << std::endl
                                                     << H_num;
}

}  // namespace state_estimation
