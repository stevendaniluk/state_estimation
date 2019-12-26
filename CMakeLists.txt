cmake_minimum_required(VERSION 3.10.0)
project(state_estimation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

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

include(CTest)
if(BUILD_TESTING MATCHES ON)

  # Try to get GTest
  find_package(GTest)
  if(NOT GTest_FOUND)
    if(STATE_ESTIMATION_BUILD_GTEST MATCHES ON)
      message("-- Will build gtest from /usr/src/googletest")
      add_subdirectory(/usr/src/googletest gtest)
    endif()
  endif()

  # Unit test for transformations utilities
  add_executable(test_transformation_utilities test/test_transformation_utilities.cpp)
  target_link_libraries(test_transformation_utilities
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_transformation_utilities)

  # Unit test for imu utilities
  add_executable(test_imu_utilities test/test_imu_utilities.cpp)
  target_link_libraries(test_imu_utilities
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_imu_utilities)

  # Unit test planer 2d state propegation model
  add_executable(test_planer_2d_state_propegation test/test_planer_2d_state_propegation.cpp)
  target_link_libraries(test_planer_2d_state_propegation
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_planer_2d_state_propegation)

  # Unit test planer 2d odometry measurement model
  add_executable(test_planer_2d_nh_odometry_measurement_model test/test_planer_2d_nh_odometry_measurement_model.cpp)
  target_link_libraries(test_planer_2d_nh_odometry_measurement_model
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_planer_2d_nh_odometry_measurement_model)

  # Unit test planer 2d imu measurement model
  add_executable(test_planer_2d_imu_measurement_model test/test_planer_2d_imu_measurement_model.cpp)
  target_link_libraries(test_planer_2d_imu_measurement_model
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_planer_2d_imu_measurement_model)

  # Unit test for the base filter
  add_executable(test_filter_base test/test_filter_base.cpp)
  target_link_libraries(test_filter_base
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_filter_base)

  # Unit test for the Kalman filter
  add_executable(test_kalman_filter test/test_kalman_filter.cpp)
  target_link_libraries(test_kalman_filter
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_kalman_filter)

  # Unit test for the EKF
  add_executable(test_ekf test/test_ekf.cpp)
  target_link_libraries(test_ekf
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_ekf)

  # Unit test for the UKF
  add_executable(test_ukf test/test_ukf.cpp)
  target_link_libraries(test_ukf
    gtest_main
    state_estimation
  )
  gtest_discover_tests(test_ukf)
endif()
