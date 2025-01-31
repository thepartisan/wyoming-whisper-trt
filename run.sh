#!/bin/bash

# Navigate to the application directory
cd /usr/src/wyoming-whisper-trt

# Activate the Python virtual environment
source .venv/bin/activate

# Check if torch2trt is installed in this venv
python -c "import torch2trt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "torch2trt not found. Installing via script/setup..."
    chmod +x script/setup
    ./script/setup
else
    echo "torch2trt is already installed. Skipping setup."
fi

# Launch the main application
python3 -m wyoming_whisper_trt \
    --model base \
    --language auto \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir /data \
    --download-dir /data \
    --device cuda \
    --debug
