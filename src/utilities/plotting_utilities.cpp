#include <state_estimation/utilities/plotting_utilities.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace state_estimation {

std::vector<Eigen::Vector2d> getEllipsePoints(double a, double b, const Eigen::Vector2d& offset,
                                              double angle, uint32_t num_pts) {
    std::vector<Eigen::Vector2d> pts(num_pts);

    // Sample the points evenly with respect to angle around the ellipse
    const Eigen::Rotation2D<double> R(angle);
    for (int i = 0; i < num_pts; ++i) {
        const double theta = i * 2.0 * M_PI / num_pts;

        const Eigen::Vector2d pos(a * cos(theta), b * sin(theta));
        pts[i] = R * pos + offset;
    }

    return pts;
}

std::vector<Eigen::Vector2d> get2DCovarianceEllipsePoints(const Eigen::Matrix2d& cov,
                                                          const Eigen::Vector2d& offset,
                                                          uint32_t num_pts) {
    // Get the eigen vectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2>> solver(cov);
    Eigen::Matrix<double, 2, 2> vectors = solver.eigenvectors();
    Eigen::Matrix<double, 2, 1> values = solver.eigenvalues();

    // The major axis corresponds to the first eigen vector, and the orientation of the ellipse is
    // the directory of the first eigen vector
    const double a = values(0);
    const double b = values(1);
    const double angle = atan2(vectors(1, 0), vectors(0, 0));
    return getEllipsePoints(a, b, offset, angle, num_pts);
}

}  // namespace state_estimation
