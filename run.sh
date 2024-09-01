# Ensure the correct working directory is set
WORKDIR /usr/src/wyoming-whisper-trt

# Run the application using the module option
CMD ["python3", "-m", "wyoming_whisper_trt", "--model", "tiny.en", "--language", "en", "--uri", "tcp://0.0.0.0:10300", "--data-dir", "/data", "--download-dir", "/data", "--debug"]
