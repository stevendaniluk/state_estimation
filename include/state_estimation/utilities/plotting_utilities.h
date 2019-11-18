#pragma once

#include <Eigen/Core>
#include <vector>

namespace state_estimation {

// getEllipsePoints
//
// Generates a set of points that form an ellipse. The points are evenly sampled with respect to
// angle around the ellipse.
//
// @param a: Minor radius (X direction)
// @param b: Major radius (Y direction)
// @param offset: Translational offset to apply to the ellipse points
// @param angle: Rotation to apply to the ellipse points
// @param num_pts: Number of points to represent the ellipse with
// @return: Ellipse coordinates
std::vector<Eigen::Vector2d> getEllipsePoints(double a, double b, const Eigen::Vector2d& offset,
                                              double angle, uint32_t num_pts = 20);

// get2DCovarianceEllipsePoints
//
// Generates a set of points describing a 2D ellipse for a 2D covariance matrix.
//
// @param cov: Coveriance matrix to generate ellipse for
// @param offset: Translational offset to position ellipse at
// @param num_pts: Number of points to represent the ellipse with
// @return: Ellipse defining the covariance matrix
std::vector<Eigen::Vector2d> get2DCovarianceEllipsePoints(const Eigen::Matrix2d& cov,
                                                          const Eigen::Vector2d& offset,
                                                          uint32_t num_pts = 20);

}  // namespace state_estimation
