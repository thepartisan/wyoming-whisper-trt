#!/usr/bin/env bash

# Activate the virtual environment and run the script
/usr/src/wyoming-whisper-trt/.venv/bin/python3 /usr/src/wyoming-whisper-trt/wyoming_whisper_trt/__main__.py --model tiny.en --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data --debug
