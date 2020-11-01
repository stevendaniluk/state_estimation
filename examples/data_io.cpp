#include "data_io.h"
#include <fstream>
#include <iostream>
#include <sstream>

bool loadCsvFields(const std::string& filename, bool skip_header,
                   const std::vector<uint32_t>& field_indices, std::vector<Eigen::VectorXd>* data) {
    std::cout << "Loading CSV file '" << filename << "'" << std::endl;
    std::ifstream file;
    file.open(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Failed to open file '" << filename << "'" << std::endl;
        return false;
    }

    std::string line;

    if (skip_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::istringstream line_sstream(line);

        // Get all the fields from this line
        std::string new_field;
        std::vector<std::string> fields;
        while (std::getline(line_sstream, new_field, ',')) {
            fields.push_back(new_field);
        }

        if (fields.empty()) {
            std::cout << "Failed to get any fields from line '" << line << "'" << std::endl;
            return false;
        }

        // Convert the field strings to floating points and save
        Eigen::VectorXd row_data(field_indices.size());
        for (int i = 0; i < field_indices.size(); ++i) {
            if (field_indices[i] >= fields.size()) {
                std::cout << "Field " << field_indices[i] << " does not exist from line '" << line
                          << "'" << std::endl;
                return false;
            }
            row_data(i) = std::stod(fields[field_indices[i]]);
        }

        data->push_back(row_data);
    }

    std::cout << "Got " << data->size() << " measurements" << std::endl;

    return true;
}

void writeToFile(const std::string& filename, const std::string& header,
                 const std::vector<Eigen::VectorXd>& data) {
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);

    std::cout << "Writing " << data.size() << " lines to file " << filename << std::endl;

    if (!header.empty()) {
        file << header + "\n";
    }
    for (const auto& line_data : data) {
        for (size_t i = 0; i < line_data.size(); ++i) {
            file << std::to_string(line_data(i));
            if (i + 1 < line_data.size()) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}
