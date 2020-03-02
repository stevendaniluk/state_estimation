#pragma once

#include <Eigen/Core>
#include <vector>

namespace state_estimation {

// getSubset
//
// @param x: Full data vector
// @param map: Which indices from x to extract, when empty all indices will be used
// @return: Ordered subset of x, size will be the number of non zero elements in map
Eigen::VectorXd getSubset(const Eigen::VectorXd& x, const std::vector<uint16_t>& map);

// getSubset
//
// @param x: Full data matrix
// @param row_map: Which rows from x to extract, when empty all indices will be used
// @param col_map: Which columns from x to extract, when empty all indices will be used
// @return: Ordered subset of x, dimension will be the number of non zero elements in row_map and
//          col_map
Eigen::MatrixXd getSubset(const Eigen::MatrixXd& x, const std::vector<uint16_t>& row_map,
                          const std::vector<uint16_t>& col_map);

// getSubset
//
// @param x: Full data matrix
// @param map: Which rows and columns (assumed to be the same) from x to extract, when empty all
//             indices will be used
// @return: Ordered subset of x, dimension will be the number of non zero elements in map
Eigen::MatrixXd getSubset(const Eigen::MatrixXd& x, const std::vector<uint16_t>& map);

// convertSubsetToFull
//
// @param subset: Vector subset to convert to full
// @param full: Full data vector to populate with subset data (remaining data will not be modified)
// @param map: Which indices in full to populate, when empty all indices will be used
void convertSubsetToFull(const Eigen::VectorXd& subset, Eigen::VectorXd* full,
                         const std::vector<uint16_t>& map);

// convertSubsetToFull
//
// @param subset: Matrix subset to convert to full
// @param full: Full data matrix to populate with subset data (remaining data will not be modified)
// @param row_map: Which rows in full to populate, when empty all indices will be used
// @param col_map: Which columns in full to populate, when empty all indices will be used
void convertSubsetToFull(const Eigen::MatrixXd& subset, Eigen::MatrixXd* full,
                         const std::vector<uint16_t>& row_map,
                         const std::vector<uint16_t>& col_map);

// convertSubsetToFull
//
// @param subset: Matrix subset to convert to full
// @param full: Full data matrix to populate with subset data (remaining data will not be modified)
// @param map: Which rows and columns in full to populate (assumed to be the same), when empty all
//             indices will be used
void convertSubsetToFull(const Eigen::MatrixXd& subset, Eigen::MatrixXd* full,
                         const std::vector<uint16_t>& map);

// convertSubsetToFullZeroed
//
// @param subset: Vector subset to convert to full
// @param map: Which indices in full to populate, when empty all indices will be used
// @param n: Size of output vector
Eigen::VectorXd convertSubsetToFullZeroed(const Eigen::VectorXd& subset,
                                          const std::vector<uint16_t>& map, uint16_t n);

// convertSubsetToFullZeroed
//
// @param subset: Matrix subset to convert to full
// @param row_map: Which rows in full to populate, when empty all indices will be used
// @param col_map: Which columns in full to populate, when empty all indices will be used
// @param m: Rows of output matrix
// @param n: Columns of output matrix
Eigen::MatrixXd convertSubsetToFullZeroed(const Eigen::MatrixXd& subset,
                                          const std::vector<uint16_t>& row_map,
                                          const std::vector<uint16_t>& col_map, uint16_t m,
                                          uint16_t n);

// convertSubsetToFullZeroed
//
// @param subset: Matrix subset to convert to full
// @param map: Which rows and columns in full to populate (assumed to be the same), when empty all
//             indices will be used
// @param m: Size of output matrix
Eigen::MatrixXd convertSubsetToFullZeroed(const Eigen::MatrixXd& subset,
                                          const std::vector<uint16_t>& map, uint16_t m);
}  // namespace state_estimation
