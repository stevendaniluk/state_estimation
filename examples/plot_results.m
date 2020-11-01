%% Load data files
clear;

% Load the filter output data
filter_data = csvread('data/filter_data.txt', 1);

filter.t = filter_data(:, 1);
filter.v = filter_data(:, 2:4);
filter.a = filter_data(:, 5:7);
filter.w = filter_data(:, 8:10);

% Load the odometry data
% (have to load as a table because it contains text fields)
odom_table = readtable('data/odom_data.txt');

odom.t = odom_table.x_time;
odom.t = (odom.t - odom.t(1)) / 1e9;
odom.v = [odom_table.field_twist_twist_linear_x, odom_table.field_twist_twist_linear_y, odom_table.field_twist_twist_linear_z];
odom.w = [odom_table.field_twist_twist_angular_x, odom_table.field_twist_twist_angular_y, odom_table.field_twist_twist_angular_z];

% Load the IMU data
% (have to load as a table because it contains text fields)
imu_table = readtable('data/imu_data.txt');

imu.t = imu_table.x_time;
imu.t = (imu.t - imu.t(1)) / 1e9;
imu.a = [imu_table.field_linear_acceleration_x, imu_table.field_linear_acceleration_y, imu_table.field_linear_acceleration_z];
imu.a(:, 3) = imu.a(:, 3) - 9.8062;
imu.w = [imu_table.field_angular_velocity_x, imu_table.field_angular_velocity_y, imu_table.field_angular_velocity_z];

%% Plotting

% Linear velocity and acceleration
figure(1);
clf;

subplot(411);
hold on;
plot(filter.t, filter.v(:, 1));
plot(filter.t, filter.v(:, 2));
plot(filter.t, filter.v(:, 3));
plot(odom.t, odom.v(:, 1));
legend({'X_E_K_F', 'Y_E_K_F', 'Z_E_K_F', 'X_o_d_o_m'});
title('Linear Velocity');
xlabel('Time [s]');
ylabel('Velocity [m/s]');

subplot(412);
hold on;
plot(filter.t, filter.a(:, 1));
plot(imu.t, imu.a(:, 1));
legend({'X_E_K_F', 'X_I_M_U'});
title('Longitudinal Acceleration');
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');

subplot(413);
hold on;
plot(filter.t, filter.a(:, 2));
plot(imu.t, imu.a(:, 2));
legend({'Y_E_K_F', 'Y_I_M_U'});
title('Lateral Acceleration');
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');

subplot(414);
hold on;
plot(filter.t, filter.a(:, 3));
plot(imu.t, imu.a(:, 3));
legend({'Z_E_K_F', 'Z_I_M_U-g'});
title('Vertical Acceleration');
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');

% Angular velocity
figure(2);
clf;

subplot(311);
hold on;
plot(filter.t, (180 / pi) * filter.w(:, 1));
plot(odom.t, (180 / pi) * odom.w(:, 1));
plot(imu.t, (180 / pi) * imu.w(:, 1));
legend({'\phi\prime_E_K_F', '\phi\prime_o_d_o_m', '\phi\prime_I_M_U'});
xlabel('Time [s]');
ylabel('Velocity [deg/s]');
title('Roll Rate');

subplot(312);
hold on;
plot(filter.t, (180 / pi) * filter.w(:, 2));
plot(odom.t, (180 / pi) * odom.w(:, 2));
plot(imu.t, (180 / pi) * imu.w(:, 2));
legend({'\theta\prime_E_K_F', '\theta\prime_o_d_o_m', '\theta\prime_I_M_U'});
xlabel('Time [s]');
ylabel('Velocity [deg/s]');
title('Pitch Rate');

subplot(313);
hold on;
plot(filter.t, (180 / pi) * filter.w(:, 3));
plot(odom.t, (180 / pi) * odom.w(:, 3));
plot(imu.t, (180 / pi) * imu.w(:, 3));
legend({'\psi\prime_E_K_F', '\psi\prime_o_d_o_m', '\psi\prime_I_M_U'});
xlabel('Time [s]');
ylabel('Velocity [deg/s]');
title('Yaw Rate');

