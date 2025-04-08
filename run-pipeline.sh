#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the required arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <input_data> <processed_data_folder> <model_folder>"
  exit 1
fi

# Step 1: Prepare the data
echo "Preparing data..."
python3 /home/administrateur/exo/mlops-simple/scripts/prepare_data.py  --input_data "$1" --output_folder "$2"

# Step 2: Train the model
echo "Training the model..."
python3 /home/administrateur/exo/mlops-simple/scripts/train_model.py  --model_folder "$3"

# Step 3: Evaluate the model
echo "Evaluating the model..."
python3 /home/administrateur/exo/mlops-simple/scripts/evaluate_model.py  --model_folder "$3" --output_folder "$2"

echo "Pipeline execution completed successfully."
