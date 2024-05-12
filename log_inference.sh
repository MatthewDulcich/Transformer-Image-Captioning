#!/bin/bash

# Ask for the run name
echo "Enter the run name:"
read run_name

# Create the directory if it doesn't exist
mkdir -p inference_logs

# Start logging into a file named after the run
log_file="inference_logs/${run_name}_run.log"
exec > >(tee "$log_file") 2>&1

# Display the contents of settings.py
echo "Contents of settings.py:"
cat settings.py
echo ""

# Run the run_inference.sh script
echo "Running run_inference.sh:"
./run_inference.sh