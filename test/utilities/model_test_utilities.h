#include <gtest/gtest.h>
#include <state_estimation/measurement_models/nonlinear_measurement_model.h>
#include <state_estimation/system_models/nonlinear_system_model.h>
#include <state_estimation/utilities/logging.h>
#include <Eigen/Core>

namespace state_estimation {

// jacobianMatchesNumericalApproximation
//
// Compares the Jacobian produced by the system model to a numerical estimate formed by perturbing
// the state
//
// @param model: Model to evaluate
// @param x: State to evaluate the Jacobian about
// @param u: Control to evaluate the Jacobian about
// @param dt: Time step to use in the model update
// @param epsilon: Amount to perturb the state variables by
// @param tolerance: Tolerance on the Frobenius norm between the numerically determine Jacobian and
//                   the output of the system model
void jacobianGMatchesNumericalApproximation(system_models::NonlinearSystemModel* model,
                                           const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                           double dt, double epsilon = 1e-6,
                                           double tolerance = 1e-3) {
    // Run our reference state through a model, then numerically compute the Jacobian by going
    // through each state variable perturbing it slightly.
    model->update(x, u, dt);
    Eigen::VectorXd x_pred = model->g();
    Eigen::MatrixXd G_target = model->G();

    Eigen::MatrixXd G_num(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i) {
        // Perturb the state slightly
        Eigen::VectorXd x_pert = x;
        x_pert(i) += epsilon;

        // Run the new state through the model
        model->update(x_pert, u, dt);

        // Compute the partial derivative
        G_num.col(i) = (model->g() - x_pred) / epsilon;
    }

    EXPECT_TRUE(G_num.isApprox(G_target, tolerance)) << "Target:" << std::endl
                                                     << printMatrix(G_target) << std::endl
                                                     << "Actual:" << std::endl
                                                     << printMatrix(G_num);
}

// jacobianVMatchesNumericalApproximation
//
// Compares the Jacobian V produced by the system model to a numerical estimate formed by perturbing
// the state
//
// @param model: Model to evaluate
// @param x: State to evaluate the Jacobian about
// @param u: Control to evaluate the Jacobian about
// @param dt: Time step to use in the model update
// @param epsilon: Amount to perturb the state variables by
// @param tolerance: Tolerance on the Frobenius norm between the numerically determine Jacobian and
//                   the output of the system model
void jacobianVMatchesNumericalApproximation(system_models::NonlinearSystemModel* model,
                                           const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                           double dt, double epsilon = 1e-6,
                                           double tolerance = 1e-3) {
    // Run our reference state through a model, then numerically compute the Jacobian by going
    // through each state variable perturbing it slightly.
    model->update(x, u, dt);
    Eigen::VectorXd x_pred = model->g();
    Eigen::MatrixXd V_target = model->V();

    Eigen::MatrixXd V_num(x.size(), u.size());
    for (int i = 0; i < u.size(); ++i) {
        // Perturb the control slightly
        Eigen::VectorXd u_pert = u;
        u_pert(i) += epsilon;

        // Run the new state through the model
        model->update(x, u_pert, dt);

        // Compute the partial derivative
        V_num.col(i) = (model->g() - x_pred) / epsilon;
    }

    EXPECT_TRUE(V_num.isApprox(V_target, tolerance)) << "Target:" << std::endl
                                                     << printMatrix(V_target) << std::endl
                                                     << "Actual:" << std::endl
                                                     << printMatrix(V_num);
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
                                                     << printMatrix(H_target) << std::endl
                                                     << "Actual:" << std::endl
                                                     << printMatrix(H_num);
}

// inactiveStatesDoNotChange
//
// Updates the model once with each state disabled and checks that the updated value for that
// state matches the update when all states are enabled.
//
// @param model: Model to evaluate
// @param x: State to evaluate about
// @param u: Control to evaluate about
// @param dt: Time delta to use
void inactiveStatesDoNotChange(system_models::NonlinearSystemModel* model, const Eigen::VectorXd& x,
                               const Eigen::VectorXd& u, double dt) {
    for (uint16_t i = 0; i < model->stateSize(); ++i) {
        // Set every state to active except for one
        std::vector<uint16_t> active_states;
        for (uint16_t j = 0; j < model->stateSize(); ++j) {
            if (j != i) {
                active_states.push_back(j);
            }
        }

        model->setActiveStates(active_states);
        model->update(x, u, dt);
        Eigen::VectorXd x_inactive = model->g();

        // The inactive state should not change
        EXPECT_EQ(x_inactive(i), x(i)) << "Index: " << i;
    }
}

// activeControlsChangeState
//
// Updates the model once with each control element individually enabled and checks that the state
// changes when compared to all controls disabled.
//
// Note, we can't test that a control influences the state because we have no knowledge of the
// relationship between states and controls, so we can only test a change from the null case.
//
// @param model: Model to evaluate
// @param x: State to evaluate about
// @param u: Control to evaluate about
// @param dt: Time delta to use
void activeControlsChangeState(system_models::NonlinearSystemModel* model, const Eigen::VectorXd& x,
                               const Eigen::VectorXd& u, double dt) {
    // Get the referece update (have to provide some dud control indices)
    model->setActiveControls({999});
    model->update(x, u, dt);
    Eigen::VectorXd x_ref = model->g();

    for (uint16_t i = 0; i < model->controlSize(); ++i) {
        // Only set one control to active
        std::vector<uint16_t> active_controls = {i};

        model->setActiveControls(active_controls);
        model->update(x, u, dt);
        Eigen::VectorXd x = model->g();

        // Something should have changed
        EXPECT_FALSE(x.isApprox(x_ref, 1e-6)) << "Index: " << i << ", Ref:" << std::endl
                                              << x_ref.transpose() << ", Updated:" << x.transpose();
    }
}

}  // namespace state_estimation
