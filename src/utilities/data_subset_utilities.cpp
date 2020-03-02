#include <assert.h>
#include <state_estimation/utilities/data_subset_utilities.h>

namespace state_estimation {

Eigen::VectorXd getSubset(const Eigen::VectorXd& x, const std::vector<uint16_t>& map) {
    if (!map.empty()) {
        Eigen::VectorXd subset(map.size());
        for (uint16_t i = 0; i < subset.size(); ++i) {
            subset(i) = x(map[i]);
        }

        return subset;
    } else {
        return x;
    }
}

Eigen::MatrixXd getSubset(const Eigen::MatrixXd& x, const std::vector<uint16_t>& row_map,
                          const std::vector<uint16_t>& col_map) {
    if (!row_map.empty() || !col_map.empty()) {
        uint16_t rows = row_map.empty() ? x.rows() : row_map.size();
        uint16_t cols = col_map.empty() ? x.cols() : col_map.size();
        Eigen::MatrixXd subset(rows, cols);
        for (uint16_t i = 0; i < rows; ++i) {
            for (uint16_t j = 0; j < cols; ++j) {
                uint16_t x_row = row_map.empty() ? i : row_map[i];
                uint16_t x_col = col_map.empty() ? j : col_map[j];
                subset(i, j) = x(x_row, x_col);
            }
        }

        return subset;
    } else {
        return x;
    }
}

Eigen::MatrixXd getSubset(const Eigen::MatrixXd& x, const std::vector<uint16_t>& map) {
    if (!map.empty()) {
        Eigen::MatrixXd subset(map.size(), map.size());
        for (uint16_t i = 0; i < subset.rows(); ++i) {
            for (uint16_t j = 0; j < subset.cols(); ++j) {
                subset(i, j) = x(map[i], map[j]);
            }
        }

        return subset;
    } else {
        return x;
    }
}

void convertSubsetToFull(const Eigen::VectorXd& subset, Eigen::VectorXd* full,
                         const std::vector<uint16_t>& map) {
    if (!map.empty()) {
        for (uint16_t i = 0; i < map.size(); ++i) {
            (*full)(map[i]) = subset(i);
        }
    } else {
        (*full) = subset;
    }
}

void convertSubsetToFull(const Eigen::MatrixXd& subset, Eigen::MatrixXd* full,
                         const std::vector<uint16_t>& row_map,
                         const std::vector<uint16_t>& col_map) {
    if (!row_map.empty() || !col_map.empty()) {
        uint16_t rows = row_map.empty() ? full->rows() : row_map.size();
        uint16_t cols = col_map.empty() ? full->cols() : col_map.size();
        for (uint16_t i = 0; i < rows; ++i) {
            for (uint16_t j = 0; j < cols; ++j) {
                uint16_t row = row_map.empty() ? i : row_map[i];
                uint16_t col = col_map.empty() ? j : col_map[j];
                (*full)(row, col) = subset(i, j);
            }
        }
    } else {
        (*full) = subset;
    }
}

void convertSubsetToFull(const Eigen::MatrixXd& subset, Eigen::MatrixXd* full,
                         const std::vector<uint16_t>& map) {
    if (!map.empty()) {
        for (uint16_t i = 0; i < map.size(); ++i) {
            for (uint16_t j = 0; j < map.size(); ++j) {
                (*full)(map[i], map[j]) = subset(i, j);
            }
        }
    } else {
        (*full) = subset;
    }
}

Eigen::VectorXd convertSubsetToFullZeroed(const Eigen::VectorXd& subset,
                                          const std::vector<uint16_t>& map, uint16_t n) {
    if (!map.empty()) {
        Eigen::VectorXd full = Eigen::VectorXd::Zero(n);
        for (uint16_t i = 0; i < map.size(); ++i) {
            full(map[i]) = subset(i);
        }

        return full;
    } else {
        return subset;
    }
}

Eigen::MatrixXd convertSubsetToFullZeroed(const Eigen::MatrixXd& subset,
                                          const std::vector<uint16_t>& row_map,
                                          const std::vector<uint16_t>& col_map, uint16_t m,
                                          uint16_t n) {
    if (!row_map.empty() || !col_map.empty()) {
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(m, n);

        uint16_t rows = row_map.empty() ? m : row_map.size();
        uint16_t cols = col_map.empty() ? n : col_map.size();
        for (uint16_t i = 0; i < rows; ++i) {
            for (uint16_t j = 0; j < cols; ++j) {
                uint16_t row = row_map.empty() ? i : row_map[i];
                uint16_t col = col_map.empty() ? j : col_map[j];
                full(row, col) = subset(i, j);
            }
        }

        return full;
    } else {
        return subset;
    }
}

Eigen::MatrixXd convertSubsetToFullZeroed(const Eigen::MatrixXd& subset,
                                          const std::vector<uint16_t>& map, uint16_t m) {
    if (!map.empty()) {
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(m, m);
        for (uint16_t i = 0; i < map.size(); ++i) {
            for (uint16_t j = 0; j < map.size(); ++j) {
                full(map[i], map[j]) = subset(i, j);
            }
        }

        return full;
    } else {
        return subset;
    }
}

}  // namespace state_estimation
