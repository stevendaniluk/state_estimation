#include <state_estimation/definitions/common_measurements.h>
#include <state_estimation/definitions/six_d_rates.h>
#include <state_estimation/filters/ekf.h>
#include <state_estimation/filters/ekf_vs.h>
#include <state_estimation/measurement_models/six_d_rates_imu.h>
#include <state_estimation/measurement_models/six_d_rates_odom.h>
#include <state_estimation/system_models/six_d_rates.h>
#include <state_estimation/utilities/integration.h>
#include "data_io.h"

using namespace state_estimation;

void usage() {
    std::cout << "This runs an example for the 6D rates estimation using an EKF with odometry and"
              << " IMU measurement inputs." << std::endl
              << "Usage: six_d_rates_example ODOM_DATA IMU_DATA OUTPUT_FILTER_DATA" << std::endl
              << "Example:" << std::endl
              << "  six_d_rates_example odom_data.txt imu_data.txt filter_data.txt" << std::endl;
}

enum class MeasTypes { ODOM, IMU };

// Helper containers for data
struct Meas {
    double t;
    Eigen::VectorXd z;
    MeasTypes type;
};
struct State {
    double t;
    Eigen::VectorXd x;
    Eigen::MatrixXd cov;
};

// Here can define functions to provide to the filter that will effectively freeze the estimate
// for instances that you know it is not changing and want any rates to be forced to zero.
//
// Each filter model can use it's own predicates. We'll define different checks for each one, but
// use the same function for zeroing the state rates across all models

bool isStationarySystem(const Eigen::VectorXd& x, const Eigen::VectorXd& z) {
    bool linear = fabs(x(six_d_rates::state::VX)) < 0.05;
    bool angular = fabs(x(six_d_rates::state::VPSI)) < 0.05;
    return linear && angular;
}

bool isStationaryOdom(const Eigen::VectorXd& x, const Eigen::VectorXd& z) {
    bool linear = fabs(x(six_d_rates::state::VX)) < 0.05 && fabs(z(meas::odom::VX)) < 0.05;
    bool angular = fabs(x(six_d_rates::state::VPSI)) < 0.05 && fabs(z(meas::odom::VPSI)) < 0.05;
    return linear && angular;
}

bool isStationaryImu(const Eigen::VectorXd& x, const Eigen::VectorXd& z) {
    bool linear = fabs(x(six_d_rates::state::VX)) < 0.05 && fabs(z(meas::imu::AX)) < 0.30;
    bool angular = fabs(x(six_d_rates::state::VPSI)) < 0.05 && fabs(z(meas::imu::VPSI)) < 0.30;
    return linear && angular;
}

void makeStationary(Eigen::VectorXd* x, Eigen::MatrixXd* cov) {
    x->segment(six_d_rates::state::VX, 3) << 0, 0, 0;
    x->segment(six_d_rates::state::AX, 3) << 0, 0, 0;
    x->segment(six_d_rates::state::VPHI, 3) << 0, 0, 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }

    // Load the odometry and imu data and sort it by timestamps, and offset all timestamps to start
    std::vector<Eigen::VectorXd> odom_data;
    std::vector<uint32_t> odom_fields = {2, 12, 13, 14, 17};
    if (!loadCsvFields(std::string(argv[1]), true, odom_fields, &odom_data)) {
        return 1;
    }
    std::vector<Eigen::VectorXd> imu_data;
    std::vector<uint32_t> imu_fields = {2, 11, 12, 13, 8, 9, 10};
    if (!loadCsvFields(std::string(argv[2]), true, imu_fields, &imu_data)) {
        return 1;
    }

    std::vector<Meas> measurements;
    for (const auto& meas_vec : odom_data) {
        Meas meas{meas_vec(0), meas_vec.tail(4), MeasTypes::ODOM};
        measurements.push_back(meas);
    }
    for (const auto& meas_vec : imu_data) {
        Meas meas{meas_vec(0), meas_vec.tail(6), MeasTypes::IMU};
        measurements.push_back(meas);
    }

    // Sort it my timestamps, offset so it starts at zero, and convert from nano seconds to
    // seconds
    std::sort(measurements.begin(), measurements.end(),
              [](const Meas& lhs, const Meas& rhs) { return lhs.t < rhs.t; });

    double t_start = measurements.front().t;
    for (auto& meas : measurements) {
        meas.t -= t_start;
        meas.t /= 1e9;
    }

    // Setup the system and measurement models
    system_models::SixDRates sys_model;
    measurement_models::SixDRatesOdom odom_meas_model;
    measurement_models::SixDRatesImu imu_meas_model;

    // Define functions to determine when the vehicle is stationary
    sys_model.setIsStationaryFunction(&isStationarySystem);
    sys_model.setMakeStationaryFunction(&makeStationary);

    odom_meas_model.setIsStationaryFunction(&isStationaryOdom);
    odom_meas_model.setMakeStationaryFunction(&makeStationary);

    imu_meas_model.setIsStationaryFunction(&isStationaryImu);
    imu_meas_model.setMakeStationaryFunction(&makeStationary);

    // The filter will estimate the state at the position of the IMU, so we need define a transform
    // from the state frame (IMU) to the odometry frame
    Eigen::Isometry3d odom_tf = Eigen::Isometry3d::Identity();
    odom_tf.translation().x() = -0.26;
    odom_meas_model.setTf(odom_tf);

    // Define our process noise
    Eigen::MatrixXd R_p = Eigen::MatrixXd::Zero(15, 15);
    R_p.diagonal().segment(0, 3) << 1e-1, 1e-2, 1e-5;
    R_p.diagonal().segment(3, 3) << 1e-7, 1e-7, 1e-7;
    R_p.diagonal().segment(6, 3) << 1e-6, 1e-6, 0.1;
    R_p.diagonal().segment(9, 3) << 1e-8, 1e-8, 1e-8;
    R_p.diagonal().segment(12, 3) << 1e-12, 1e-12, 1e-12;
    sys_model.setProcessCovariance(R_p);

    // Define our measurement covariances
    Eigen::MatrixXd cov_odom = Eigen::MatrixXd::Zero(meas::odom::DIMS, meas::odom::DIMS);
    cov_odom.diagonal() << 1e-5, 1e-4, 1e-6, 5e-1;

    Eigen::MatrixXd cov_imu = Eigen::MatrixXd::Zero(meas::imu::DIMS, meas::imu::DIMS);
    cov_imu.diagonal() << 1e-3, 1e-3, 1e-2, 1e-1, 1e-1, 0.0001;

    // Setup the filter initial states
    double t_i = measurements.front().t;
    Eigen::VectorXd x_i = Eigen::VectorXd::Zero(sys_model.stateSize());
    x_i(six_d_rates::state::GZ) = 9.8062;

    Eigen::MatrixXd cov_i = Eigen::MatrixXd::Zero(sys_model.stateSize(), sys_model.stateSize());
    cov_i.diagonal() = 1e-2 * Eigen::VectorXd::Ones(sys_model.stateSize());
    cov_i.diagonal().segment(six_d_rates::state::VX, 3) << 1e-5, 1e-5, 1e-5;
    cov_i.diagonal().segment(six_d_rates::state::AX, 3) << 1e-4, 1e43, 1e-4;
    cov_i.diagonal().segment(six_d_rates::state::VPHI, 3) << 1e-5, 1e-5, 1e-5;
    cov_i.diagonal().segment(six_d_rates::state::GX, 3) << 1e-9, 1e-9, 1e-9;
    cov_i.diagonal().segment(six_d_rates::state::B_AX, 3) << 1e-6, 1e-6, 1e-6;
    cov_i.diagonal().segment(six_d_rates::state::B_WX, 3) << 1e-3, 1e-3, 1e-3;

    // Here we'll create the actual filter. We'll use a variable state version which can disable
    // some states from being estimated. If you want the full state estimated use the non variable
    // version:
    //   EKF filter(&sys_model, x_i, cov_i, t_i);

    EKFVS filter(&sys_model, x_i, cov_i, t_i);

    std::vector<uint16_t> active_states;
    active_states.push_back(six_d_rates::state::VX);
    active_states.push_back(six_d_rates::state::VY);
    active_states.push_back(six_d_rates::state::VZ);
    active_states.push_back(six_d_rates::state::AX);
    active_states.push_back(six_d_rates::state::AY);
    active_states.push_back(six_d_rates::state::AZ);
    active_states.push_back(six_d_rates::state::VPHI);
    active_states.push_back(six_d_rates::state::VTHETA);
    active_states.push_back(six_d_rates::state::VPSI);
    sys_model.setActiveStates(active_states);

    std::vector<uint16_t> active_odom;
    active_odom.push_back(meas::odom::VX);
    active_odom.push_back(meas::odom::VY);
    active_odom.push_back(meas::odom::VZ);
    active_odom.push_back(meas::odom::VPSI);
    odom_meas_model.setActiveMeasurements(active_odom);

    std::vector<uint16_t> active_imu;
    active_imu.push_back(meas::imu::AX);
    active_imu.push_back(meas::imu::AY);
    active_imu.push_back(meas::imu::AZ);
    active_imu.push_back(meas::imu::VPHI);
    active_imu.push_back(meas::imu::VTHETA);
    active_imu.push_back(meas::imu::VPSI);
    imu_meas_model.setActiveMeasurements(active_imu);

    // Run through the measurements, saving the state and covarience after each update
    std::vector<State> filter_states;
    Eigen::Vector3d pos_integrated = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_integrated = Eigen::Quaterniond::Identity();
    int count = 0;
    for (const auto& meas : measurements) {
        // double t_i = filter.getStateTime();
        if (meas.type == MeasTypes::ODOM) {
            filter.correct(meas.z, cov_odom, meas.t, &odom_meas_model);
        } else if (meas.type == MeasTypes::IMU) {
            filter.correct(meas.z, cov_imu, meas.t, &imu_meas_model);
        }

        State state = {filter.getStateTime(), filter.getState(), filter.getCovariance()};
        filter_states.push_back(state);
    }

    // Write the state data to a file
    std::vector<Eigen::VectorXd> states_csv;
    for (const auto& sample : filter_states) {
        uint32_t index = 0;
        uint32_t n = six_d_rates::state::DIMS;
        Eigen::VectorXd data(1 + n + n * n);

        data(index) = sample.t;
        index++;

        data.segment(index, n) = sample.x;
        index += n;

        for (int i = 0; i < n; ++i) {
            data.segment(index, n) = sample.cov.row(i);
            index += n;
        }

        states_csv.push_back(data);
    }

    std::string header =
        "t,VX,VY,VZ,AX,AY,AZ,VPHI,VTHETA,VPSI,GX,GY,GZ,B_AX,B_AY,B_AZ,B_WX,B_WY,B_WZ";
    for (int i = 0; i < six_d_rates::state::DIMS; ++i) {
        for (int j = 0; j < six_d_rates::state::DIMS; ++j) {
            header += ",cov_" + std::to_string(i) + std::to_string(j);
        }
    }

    writeToFile(std::string(argv[3]), header, states_csv);

    return 0;
}
