"""Event handler for clients of the server."""

import argparse
import asyncio
import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import torch
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

import whisper_trt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Formats the time with nanosecond precision."""
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


# Set up logging with the custom formatter
formatter = NanosecondFormatter("%(asctime)s [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = [handler]


class WhisperTrtEventHandler(AsyncEventHandler):
    """Event handler for clients utilizing the Whisper TRT model."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: whisper_trt.WhisperTRT,
        model_lock: asyncio.Lock,
        initial_prompt: Optional[str] = None,
        *args,
        model_is_lang_specific: bool = False,
        default_language: str = "auto",
        **kwargs,
    ) -> None:
        """
        Initializes the WhisperTrtEventHandler.

        Args:
            reader (asyncio.StreamReader): The reader stream from the client connection.
            writer (asyncio.StreamWriter): The writer stream to the client connection.
            wyoming_info (Info): Information about the Wyoming server.
            cli_args (argparse.Namespace): Command-line arguments.
            model (whisper_trt.WhisperTRT): The Whisper TRT model instance.
            model_lock (asyncio.Lock): Asynchronous lock for model access.
            initial_prompt (Optional[str], optional): Initial prompt for transcription. Defaults to None.
            model_is_lang_specific (bool, optional): Indicates if the loaded model is language-specific.
            default_language (str, optional): Default language to use for transcription. Defaults to "auto".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Remove extra arguments so the base class won't receive them.
        if "model_is_lang_specific" in kwargs:
            del kwargs["model_is_lang_specific"]
        if "default_language" in kwargs:
            del kwargs["default_language"]

        super().__init__(reader, writer, *args, **kwargs)  # Initialize the base class

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt

        # Language and model-specific flags
        self.model_is_lang_specific = model_is_lang_specific
        self.default_language = default_language

        # Start with language from CLI or the default
        self._language = self.cli_args.language if hasattr(self.cli_args, "language") else self.default_language
        
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = Path(self._wav_dir.name) / "speech.wav"
        self._wave_writer: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        """Handles incoming events from clients."""
        try:
            if AudioChunk.is_type(event.type):
                await self._handle_audio_chunk(event)
                return True
            if AudioStop.is_type(event.type):
                await self._handle_audio_stop()
                return False
            if Transcribe.is_type(event.type):
                await self._handle_transcribe(event)
                return True
            if Describe.is_type(event.type):
                await self._handle_describe()
                return True
            return True
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}")
            return True  # Continue handling other events

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Handles an AudioChunk event by writing audio data to a WAV file."""
        chunk = AudioChunk.from_event(event)

        if self._wave_writer is None:
            try:
                self._wave_writer = wave.open(str(self._wav_path), "wb")
                self._wave_writer.setframerate(chunk.rate)
                self._wave_writer.setsampwidth(chunk.width)
                self._wave_writer.setnchannels(chunk.channels)
                logger.debug(f"Initialized WAV file at '{self._wav_path}'.")
            except wave.Error as e:
                logger.error(f"Failed to open WAV file: {e}")
                raise

        self._wave_writer.writeframes(chunk.audio)
        logger.debug(f"Wrote {len(chunk.audio)} frames to WAV file.")

    async def _handle_audio_stop(self) -> None:
        """Handles an AudioStop event by transcribing the recorded audio."""
        if self._wave_writer is None:
            logger.warning("AudioStop received but no audio was recorded.")
            return
        try:
            self._wave_writer.close()
            logger.debug(f"Closed WAV file at '{self._wav_path}'.")
        except wave.Error as e:
            logger.error(f"Failed to close WAV file: {e}")
            raise
        finally:
            self._wave_writer = None

        async with self.model_lock:
            try:
                # Transcribe using the currently set language.
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.transcribe, str(self._wav_path), self._language
                )
                text = result.get("text", "")
                logger.info(f"Transcription result: {text}")
                await self.write_event(Transcript(text=text).event())
                logger.debug("Completed transcription request.")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                await self.write_event(Transcript(text="Transcription failed.").event())

        # Reset language to CLI argument or default
        self._language = self.cli_args.language if hasattr(self.cli_args, "language") else self.default_language
        logger.debug(f"Reset language to '{self._language}'.")

    async def _handle_transcribe(self, event: Event) -> None:
        """Handles a Transcribe event by setting the language."""
        transcribe = Transcribe.from_event(event)
        if transcribe.language:
            self._language = transcribe.language
            logger.debug(f"Language set to '{transcribe.language}'.")

    async def _handle_describe(self) -> None:
        """Handles a Describe event by sending server information."""
        try:
            await self.write_event(self.wyoming_info_event)
            logger.debug("Sent server information.")
        except Exception as e:
            logger.error(f"Failed to send server information: {e}")

    def cleanup(self) -> None:
        """Cleans up resources such as temporary directories."""
        if self._wav_dir:
            self._wav_dir.cleanup()
            logger.debug(f"Cleaned up temporary directory '{self._wav_dir.name}'.")
