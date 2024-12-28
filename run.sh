#!/bin/bash
source /usr/src/wyoming-whisper-trt/.venv/bin/activate

# Navigate to the application directory
cd /usr/src/wyoming-whisper-trt && python3 -m wyoming_whisper_trt --model base.en --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data --device cuda --debug
