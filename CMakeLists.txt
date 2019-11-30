cmake_minimum_required(VERSION 2.8.3)
project(state_estimation)

find_package(Eigen3 REQUIRED)

option(STATE_ESTIMATION_BUILD_TEST "Build the state_estimation unit tests" ON)
option(STATE_ESTIMATION_BUILD_GTEST "Build google test library for unit tests" OFF)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  include
)

###########
## Build ##
###########

add_library(state_estimation
  src/filters/ekf.cpp
  src/filters/filter_base.cpp
  src/filters/kalman_filter.cpp
  src/filters/ukf.cpp
  src/measurement_models/linear_measurement_model.cpp
  src/measurement_models/measurement_model.cpp
  src/measurement_models/nonlinear_measurement_model.cpp
  src/measurement_models/planer_2d_imu.cpp
  src/measurement_models/planer_2d_nh_odometry.cpp
  src/system_models/linear_system_model.cpp
  src/system_models/nonlinear_system_model.cpp
  src/system_models/planer_2d_state_propegation.cpp
  src/system_models/system_model.cpp
  src/utilities/imu_utilities.cpp
  src/utilities/plotting_utilities.cpp
  src/utilities/transformation_utilities.cpp
)

# Install the state estimation library and header files
install(TARGETS
  state_estimation
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
install(DIRECTORY
  include/state_estimation
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

#############
## Testing ##
#############
if(STATE_ESTIMATION_BUILD_TEST MATCHES ON)
  if(STATE_ESTIMATION_BUILD_GTEST MATCHES ON)
    add_subdirectory(/usr/src/googletest gtest)
  endif()

  # Unit test for transformations utilities
  add_executable(test_transformation_utilities test/test_transformation_utilities.cpp)
  add_test(NAME test_transformation_utilities COMMAND test_transformation_utilities)
  target_link_libraries(test_transformation_utilities
    gtest_main
    state_estimation
  )

  # Unit test for imu utilities
  add_executable(test_imu_utilities test/test_imu_utilities.cpp)
  add_test(NAME test_imu_utilities COMMAND test_imu_utilities)
  target_link_libraries(test_imu_utilities
    gtest_main
    state_estimation
  )

  # Unit test planer 2d state propegation model
  add_executable(test_planer_2d_state_propegation test/test_planer_2d_state_propegation.cpp)
  add_test(NAME test_planer_2d_state_propegation COMMAND test_planer_2d_state_propegation)
  target_link_libraries(test_planer_2d_state_propegation
    gtest_main
    state_estimation
  )

  # Unit test planer 2d odometry measurement model
  add_executable(test_planer_2d_nh_odometry_measurement_model test/test_planer_2d_nh_odometry_measurement_model.cpp)
  add_test(NAME test_planer_2d_nh_odometry_measurement_model COMMAND test_planer_2d_nh_odometry_measurement_model)
  target_link_libraries(test_planer_2d_nh_odometry_measurement_model
    gtest_main
    state_estimation
  )

  # Unit test planer 2d imu measurement model
  add_executable(test_planer_2d_imu_measurement_model test/test_planer_2d_imu_measurement_model.cpp)
  add_test(NAME test_planer_2d_imu_measurement_model COMMAND test_planer_2d_imu_measurement_model)
  target_link_libraries(test_planer_2d_imu_measurement_model
    gtest_main
    state_estimation
  )

  # Unit test for the base filter
  add_executable(test_filter_base test/test_filter_base.cpp)
  add_test(NAME test_filter_base COMMAND test_filter_base)
  target_link_libraries(test_filter_base
    gtest_main
    state_estimation
  )

  # Unit test for the Kalman filter
  add_executable(test_kalman_filter test/test_kalman_filter.cpp)
  add_test(NAME test_kalman_filter COMMAND test_kalman_filter)
  target_link_libraries(test_kalman_filter
    gtest_main
    state_estimation
  )

  # Unit test for the EKF
  add_executable(test_ekf test/test_ekf.cpp)
  add_test(NAME test_ekf COMMAND test_ekf)
  target_link_libraries(test_ekf
    gtest_main
    state_estimation
  )

  # Unit test for the UKF
  add_executable(test_ukf test/test_ukf.cpp)
  add_test(NAME test_ukf COMMAND test_ukf)
  target_link_libraries(test_ukf
    gtest_main
    state_estimation
  )
endif()
