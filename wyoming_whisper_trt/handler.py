import argparse
import asyncio
import io
import logging
import time
import wave
from typing import Optional

import numpy as np
from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptStart,
    TranscriptChunk,
    TranscriptStop,
)
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

import whisper_trt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# bytes threshold for sliding-window interim decoding (0.25s @16kHz, 16-bit stereo)
PARTIAL_THRESHOLD = 16000


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


# Set up logging
formatter = NanosecondFormatter("%(asctime)s [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = [handler]


def wav_bytes_to_np_array(wav_bytes: bytes) -> np.ndarray:
    """
    Read a WAV file from an in-memory bytes object and return a NumPy array of samples.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
        audio = np.frombuffer(raw, dtype=dtype)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())
        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float32) / float(2 ** (8 * sw - 1))
        return audio


class WhisperTrtEventHandler(AsyncEventHandler):
    """Event handler for clients utilizing the Whisper TRT model with deduped interim chunks."""

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
        # remove extra kwargs
        kwargs.pop("model_is_lang_specific", None)
        kwargs.pop("default_language", None)
        super().__init__(reader, writer, *args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self._language = getattr(cli_args, "language", None) or default_language

        # state for sliding-window interim
        self._pcm_buffer = bytearray()
        self._wav_buffer = io.BytesIO()
        self._wave_writer: Optional[wave.Wave_write] = None
        self._sample_width: Optional[int] = None
        self._channels: Optional[int] = None

        self._partial_threshold = PARTIAL_THRESHOLD
        self._last_sent_chunk = 0
        self._sent_start = False
        self._last_emitted_text = ""

    async def handle_event(self, event: Event) -> bool:
        logger.debug("Received event: %s", event.type)
        if AudioStart.is_type(event.type):
            # reset per-utterance state
            start = AudioStart.from_event(event)
            logger.debug(
                "AudioStart: rate=%d, width=%d, channels=%d",
                start.rate,
                start.width,
                start.channels,
            )
            self._pcm_buffer.clear()
            self._wav_buffer = io.BytesIO()
            self._wave_writer = None
            self._sample_width = None
            self._channels = None
            self._last_sent_chunk = 0
            self._sent_start = False
            self._last_emitted_text = ""
            return True

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

    async def _handle_audio_chunk(self, event: Event) -> None:
        chunk = AudioChunk.from_event(event)

        # init WAV writer on first chunk
        if self._wave_writer is None:
            self._wav_buffer = io.BytesIO()
            self._wave_writer = wave.open(self._wav_buffer, "wb")
            self._wave_writer.setframerate(chunk.rate)
            self._wave_writer.setsampwidth(chunk.width)
            self._wave_writer.setnchannels(chunk.channels)
            self._sample_width = chunk.width
            self._channels = chunk.channels
            logger.debug(
                "Initialized WAV buffer; threshold %d bytes", self._partial_threshold
            )

        # accumulate raw PCM
        self._wave_writer.writeframes(chunk.audio)
        self._pcm_buffer.extend(chunk.audio)

        # emit TranscriptStart once
        if not self._sent_start:
            await self.write_event(TranscriptStart(language=self._language).event())
            logger.debug("➡️ Emitted TranscriptStart")
            self._sent_start = True

        # sliding window
        window = self._partial_threshold
        hop = window // 2

        # trim if too large
        max_buf = window * 10
        if len(self._pcm_buffer) > max_buf:
            del self._pcm_buffer[: len(self._pcm_buffer) - window * 2]
            logger.debug("Trimmed PCM buffer to %d bytes", len(self._pcm_buffer))

        # process windows
        while len(self._pcm_buffer) >= window:
            raw = self._pcm_buffer[:window]
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[self._sample_width]
            pcm = np.frombuffer(raw, dtype=dtype)
            if self._channels and self._channels > 1:
                pcm = pcm.reshape(-1, self._channels)
            if pcm.dtype != np.float32:
                pcm = pcm.astype(np.float32) / float(2 ** (8 * self._sample_width - 1))

            # transcribe
            loop = asyncio.get_event_loop()
            async with self.model_lock:
                result = await loop.run_in_executor(
                    None,
                    lambda pcm=pcm: self.model.transcribe(
                        pcm, self._language, stream=True
                    ),
                )

            # dedupe and emit only new, non-empty chunks
            for text in result.get("chunks", [])[self._last_sent_chunk :]:
                clean = text.strip()
                if clean and clean != self._last_emitted_text:
                    await self.write_event(TranscriptChunk(text=clean).event())
                    logger.debug("➡️ Emitted TranscriptChunk: %r", clean)
                    self._last_emitted_text = clean
            self._last_sent_chunk = len(result.get("chunks", []))

            # slide
            del self._pcm_buffer[:hop]

    async def _handle_audio_stop(self) -> None:
        logger.debug("AudioStop received; flushing final transcript")
        if self._wave_writer:
            self._wave_writer.close()
            self._wave_writer = None

        # final flush
        wav_bytes = self._wav_buffer.getvalue()
        audio_np = wav_bytes_to_np_array(wav_bytes)
        loop = asyncio.get_event_loop()
        async with self.model_lock:
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(audio_np, self._language, stream=True),
            )

        # emit any remaining interim
        for text in result.get("chunks", []):
            clean = text.strip()
            if clean and clean != self._last_emitted_text:
                await self.write_event(TranscriptChunk(text=clean).event())
                logger.debug("➡️ Final interim: %r", clean)
                self._last_emitted_text = clean

        # final transcript
        final_text = result.get("text", "").strip()
        await self.write_event(Transcript(text=final_text).event())
        logger.debug("➡️ Emitting final Transcript: %r", final_text)
        await self.write_event(TranscriptStop().event())

        # reset
        self._sent_start = False
        self._last_sent_chunk = 0
        self._last_emitted_text = ""
        self._wav_buffer = io.BytesIO()

    async def _handle_transcribe(self, event: Event) -> None:
        tr = Transcribe.from_event(event)
        if tr.language:
            self._language = tr.language
            logger.debug("Language set to: %s", self._language)

    async def _handle_describe(self) -> None:
        await self.write_event(self.wyoming_info_event)

    def cleanup(self) -> None:
        if self._wav_buffer:
            self._wav_buffer.close()
