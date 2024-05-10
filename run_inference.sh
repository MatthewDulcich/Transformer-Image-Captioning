#!/bin/bash

for img in examples_img/*.jpg; do
    echo "Processing $img"
    output=$(python -W ignore inference.py --image="$img" 2>/dev/null)
    echo "Predicted caption for $img: $output"
done