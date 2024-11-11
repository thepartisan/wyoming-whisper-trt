#!/usr/bin/env python3
"""
Main entry point for the Whisper TRT server.

This script initializes the Whisper TRT model, sets up the server, and handles client events.
"""

import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Optional, List

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import WhisperTrtEventHandler
from whisper_trt.cache import get_cache_dir, make_cache_dir
from whisper_trt.utils import check_file_md5, download_file

from whisper_trt import load_trt_model, WhisperTRT

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Formats the time with nanosecond precision."""
        ct = record.created
        t = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else ct
        s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ct))
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


def setup_logging(debug: bool, log_format: str) -> None:
    """
    Sets up logging with the specified level and format.

    Args:
        debug (bool): Whether to enable DEBUG level logging.
        log_format (str): Format string for log messages.
    """
    formatter = NanosecondFormatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers = [handler]

    logger.debug("Logging has been configured.")


def normalize_model_name(model_name: str) -> str:
    """
    Normalizes the model name to ensure it matches the expected format.

    Args:
        model_name (str): The input model name.

    Returns:
        str: The normalized model name.
    """
    mapping = {
        "tiny": "tiny.en",
        "base": "base.en",
        "small": "small.en",
        "medium": "medium.en",
        "large": "large.en"
    }
    normalized_name = mapping.get(model_name.lower(), model_name)
    logger.debug(f"Normalized model name: '{model_name}' to '{normalized_name}'.")
    return normalized_name


def extract_languages(tokenizer: WhisperTRT) -> List[str]:
    """
    Extracts supported languages from the tokenizer.

    Args:
        tokenizer (WhisperTRT): The Whisper TRT model instance.

    Returns:
        List[str]: A list of supported language codes.
    """
    try:
        languages = tokenizer.tokenizer.get_languages()  # Replace with the correct method if different
        logger.debug(f"Supported languages retrieved: {languages}")
    except AttributeError:
        logger.warning("Tokenizer does not have 'get_languages' method. Defaulting to ['en'].")
        languages = ['en']
    except Exception as e:
        logger.error(f"Error retrieving languages from tokenizer: {e}. Defaulting to ['en'].")
        languages = ['en']
    return languages


def build_wyoming_info(model_name: str, languages: List[str]) -> Info:
    """
    Builds the Wyoming Info object with ASR model details.

    Args:
        model_name (str): The name of the ASR model.
        languages (List[str]): Supported languages for the model.

    Returns:
        Info: The constructed Info object.
    """
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
                        languages=languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )
    logger.debug(f"Wyoming Info built with model '{model_name}' and languages {languages}.")
    return wyoming_info


async def run_server(
    uri: str,
    handler_class: WhisperTrtEventHandler,
    *args,
    **kwargs
) -> None:
    """
    Initializes and runs the asynchronous server.

    Args:
        uri (str): The URI to bind the server to.
        handler_class (WhisperTrtEventHandler): The event handler class.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    try:
        server = AsyncServer.from_uri(uri)
        logger.info(f"Server initialized and listening on {uri}.")
    except Exception as e:
        logger.error(f"Failed to initialize server with URI '{uri}': {e}")
        raise

    try:
        await server.run(handler_class, *args, **kwargs)
    except Exception as e:
        logger.error(f"Server encountered an error: {e}")
        raise
    finally:
        await server.close()
        logger.info("Server has been shut down.")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Whisper TRT ASR Server")
    parser.add_argument(
        "--model",
        required=True,
        help="Name of Whisper model to use (e.g., 'tiny.en', 'base.en', 'small.en')",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="Server URI to bind to (e.g., 'unix://', 'tcp://')",
    )
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
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        choices=["default", "float16", "int8"],
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    parser.add_argument(
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        help="Format for log messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version and exit",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Default language to set for transcription",
    )

    args = parser.parse_args()

    # Set download directory to the first data directory if not specified
    if not args.download_dir:
        args.download_dir = args.data_dir[0]
        logger.debug(f"No download directory specified. Using first data directory: {args.download_dir}")

    # Setup logging
    setup_logging(args.debug, args.log_format)

    # Normalize model name
    model_name = normalize_model_name(args.model)

    # Ensure the download directory exists
    download_path = Path(args.download_dir)
    try:
        download_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured download directory exists at '{download_path}'.")
    except OSError as e:
        logger.error(f"Failed to create download directory at '{download_path}': {e}")
        sys.exit(1)

    # Load Whisper TRT model
    try:
        logger.info(f"Loading Whisper TRT model '{model_name}'...")
        whisper_model = load_trt_model(model_name, path=str(download_path / f"{model_name}.pth"), build=True)
        logger.info(f"Whisper TRT model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper TRT model '{model_name}': {e}")
        sys.exit(1)

    # Extract supported languages from the tokenizer
    languages = extract_languages(whisper_model)

    # Build Wyoming Info
    wyoming_info = build_wyoming_info(model_name, languages)

    # Initialize asyncio lock for model access
    model_lock = asyncio.Lock()
    logger.debug("Initialized asyncio lock for model access.")

    # Initialize the event handler
    event_handler = partial(
        WhisperTrtEventHandler,
        wyoming_info=wyoming_info,
        cli_args=args,
        model=whisper_model,
        model_lock=model_lock,
        initial_prompt=args.initial_prompt,
    )

    # Run the server
    try:
        logger.info("Starting the Whisper TRT ASR server...")
        await run_server(args.uri, event_handler)
    except Exception as e:
        logger.error(f"Server encountered an unexpected error: {e}")
        sys.exit(1)


def run() -> None:
    """Runs the main asynchronous entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
