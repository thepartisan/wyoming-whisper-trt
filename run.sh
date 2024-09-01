#!/bin/bash

# Navigate to the application directory
cd /usr/src/wyoming-whisper-trt/wyoming_whisper_trt && python3 __main__.py --model tiny.en --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data --debug