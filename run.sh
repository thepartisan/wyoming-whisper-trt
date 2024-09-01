#!/usr/bin/env bash

# Activate the virtual environment
source /usr/src/wyoming-whisper-trt/.venv/bin/activate

# Run the script with the appropriate arguments
/usr/src/wyoming-whisper-trt/script/run --model tiny.en --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data --debug
