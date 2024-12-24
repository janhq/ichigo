#!/bin/bash
# Path to your Python script
PYTHON_SCRIPT="audio_to_audio_tokens.py"

# Path to your config file
CONFIG_FILE="configs/audio_to_audio_tokens_cfg.yaml"

# Loop from 15 to 24
for id in {0..23}
do
    echo "Processing batch $id"

    name="/home/jan/BachVD/audio_data/mls_eng_10k/output_${id}/"
    
    # Construct the save_dir path
    SAVE_DIR="/home/jan/BachVD/audio_data/raw_audio_token/output_${id}/"

    
    # Run the Python script with the constructed paths
    python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$name" --save_dir="$SAVE_DIR" 
    
    echo "Finished processing batch $id"
    echo "------------------------"
done

echo "All batches processed"