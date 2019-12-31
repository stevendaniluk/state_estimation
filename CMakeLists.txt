cmake_minimum_required(VERSION 3.10.0)
project(state_estimation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

option(DEBUG_STATE_ESTIMATION "Print debug logs" OFF)
if(DEBUG_STATE_ESTIMATION)
  add_definitions(-DDEBUG_STATE_ESTIMATION)
endif()

option(STATE_ESTIMATION_BUILD_GTEST "Build google test library for unit tests" OFF)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  include
)

###########
## Build ##
###########

file(GLOB SOURCE_FILES
    src/definitions/*.cpp
    src/filters/*.cpp
    src/measurement_models/*.cpp
    src/system_models/*.cpp
    src/utilities/*.cpp
)

add_library(state_estimation
    ${SOURCE_FILES}
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

  # Define all the tests we want to build, the loop below will create them all
  set(TEST_TARGETS
    test_angle_utilities
    test_transformation_utilities
    test_imu_utilities
    test_planer_2d_state_propegation
    test_planer_2d_nh_odometry_measurement_model
    test_planer_2d_imu_measurement_model
    test_filter_base
    test_kalman_filter
    test_ekf
    test_ukf
  )

  foreach(target ${TEST_TARGETS})
    add_executable(${target} test/${target}.cpp)
    target_link_libraries(${target}
      gtest_main
      state_estimation
    )
    gtest_discover_tests(${target})
  endforeach()

endif()
