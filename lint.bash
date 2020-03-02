#!/usr/bin/env bash

# All the folders we want to lint
folders=(
    include/state_estimation/definitions
    include/state_estimation/filters
    include/state_estimation/measurement_models
    include/state_estimation/system_models
    include/state_estimation/utilities
    src/filters
    src/measurement_models
    src/system_models
    src/utilities
    test
)

# Get the root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for folder in "${folders[@]}"; do
    for filename in $DIR/$folder/*.{h,hpp,cpp}; do
        if [ -f "$filename" ]; then
            echo "Formatting "$filename
            clang-format -i $filename
        fi
    done
done
