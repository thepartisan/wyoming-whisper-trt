#!/usr/bin/env bash
cd /usr/src/ && 
python3 wyoming-whisper-trt/script/run --model tiny.en --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data