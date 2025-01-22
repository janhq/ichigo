# Path to your Python script
PYTHON_SCRIPT="audio_to_audio_tokens.py"

# Path to your config file
CONFIG_FILE="configs/audio_to_audio_tokens_cfg.yaml"

echo "Processing data"
NAME="/home/root/BachVD/Audio_data/libritts_r_filtered/clean/"
# Construct the save_dir path
SAVE_DIR="/home/root/BachVD/Audio_data/libritts_r_filtered/ichigo_tokens_v2/"


# Run the Python script with the constructed paths
python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$NAME"  --save_dir="$SAVE_DIR" 

echo "Finished processing data"
echo "------------------------"