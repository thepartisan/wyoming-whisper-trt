#!/usr/bin/env bash
cd /usr/src/ && 
python3 ./script/run --model tiny --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data