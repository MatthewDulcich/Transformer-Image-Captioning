#!/bin/bash

# This script runs the inference.py script on all .jpg images in the examples_img directory.
# It iterates over each .jpg image in the directory, runs the inference script on the image,
# and prints the predicted caption for the image.
#
# The script suppresses all warnings from the python script and redirects stderr to /dev/null
# to prevent error messages from being printed to the console.
#
# Usage:
#   ./run_inference.sh
#
# Requirements:
#   - The inference.py script must be in the same directory as this script.
#   - The examples_img directory must exist and contain .jpg images.

for img in examples_img/*.jpg; do
    echo "Processing $img"
    output=$(python -W ignore inference.py --image="$img" 2>/dev/null)
    echo "Predicted caption for $img: $output"
done