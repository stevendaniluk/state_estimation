#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

// loadCsvFields
//
// Loads floating point values from a CSV file.
//
// @param filename: Path to file to load
// @param skip_header: If the first line should be skipped
// @param field_indices: Which fields to load ([0, 2, 5] will load the first, third, and 6th fields)
// @param data: Data vector to populate
bool loadCsvFields(const std::string& filename, bool skip_header,
                   const std::vector<uint32_t>& field_indices, std::vector<Eigen::VectorXd>* data);

// writeToFile
//
// This will overwrite the contents of the file if it already exists.
//
// @param filename: File to write to
// @param header: First line to write, skipped when empty
// @param data: Data elements to write
void writeToFile(const std::string& filename, const std::string& header,
                 const std::vector<Eigen::VectorXd>& data);
