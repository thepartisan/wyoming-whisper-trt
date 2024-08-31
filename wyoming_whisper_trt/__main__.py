#!/usr/bin/env python3
import argparse
import asyncio
import logging
import re

from functools import partial
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import WhisperTrtEventHandler

from whisper_trt import load_trt_model

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of whisper model to use (e.g., 'tiny.en', 'base.en', 'small.en')",
        nargs='?',
        const="tiny.en"
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    parser.add_argument(
        "--language",
        help="Default language to set for transcription",
        type=str,
        default="en"
    )

    args = parser.parse_args()

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    model_name = args.model

    # Ensure the model name is supported by WhisperTRT
    _LOGGER.debug("Checking model name: %s", model_name)
    if model_name == "tiny":
        model_name = "tiny.en"
    elif model_name == "base":
        model_name = "base.en"
    elif model_name == "small":
        model_name = "small.en"
    _LOGGER.debug("Using model name: %s", model_name)

    try:
        # Load WhisperTRT model
        _LOGGER.debug("Loading model: %s", model_name)
        whisper_model = load_trt_model(model_name)
        _LOGGER.debug("Model loaded successfully")
    except Exception as e:
        _LOGGER.error("Failed to load model: %s", e)
        raise

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper-trt",
                description="OpenAI Whisper with TensorRT",
                attribution=Attribution(
                    name="NVIDIA-AI-IOT",
                    url="https://github.com/NVIDIA-AI-IOT/whisper_trt",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://huggingface.co/OpenAI",
                        ),
                        installed=True,
                        languages=whisper_model.tokenizer.LANGUAGE_CODES,  # Use tokenizer's language codes
                        version=__version__,
                    )
                ],
            )
        ],
    )

    try:
        # Initialize the server
        _LOGGER.debug("Initializing server with URI: %s", args.uri)
        server = AsyncServer.from_uri(args.uri)
        _LOGGER.debug("Server initialized successfully")
    except Exception as e:
        _LOGGER.error("Failed to initialize server: %s", e)
        raise

    _LOGGER.info("Server ready and waiting for connections")
    
    model_lock = asyncio.Lock()
    try:
        await server.run(
            partial(
                WhisperTrtEventHandler,
                wyoming_info,
                args,
                whisper_model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    except Exception as e:
        _LOGGER.error("Error while running the server: %s", e)
        raise

# -----------------------------------------------------------------------------

def run() -> None:
    try:
        asyncio.run(main())
    except Exception as e:
        _LOGGER.error("Unhandled exception: %s", e)
        raise

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down due to keyboard interrupt")
