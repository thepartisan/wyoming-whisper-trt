# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import argparse
import os
import psutil
import re
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch2trt
import tensorrt

from whisper import load_model
from whisper.model import LayerNorm, Linear, Tensor, ModelDimensions, sinusoids, Whisper
from whisper.tokenizer import Tokenizer, LANGUAGES, TO_LANGUAGE_CODE
import whisper.audio
from dataclasses import asdict

from .cache import get_cache_dir, make_cache_dir
from .__version__ import __version__

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# AUDIO ENCODER
# -----------------------------------------------------------------------------


class _AudioEncoderEngine(nn.Module):
    """
    Audio Encoder Engine for Whisper TRT.

    This module allows for online substitution of the positional embedding.
    (This implementation preserves the original logic.)
    """

    def __init__(
        self, conv1: nn.Conv1d, conv2: nn.Conv1d, blocks: nn.Module, ln_post: LayerNorm
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.blocks = blocks
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x: Tensor, positional_embedding: Tensor) -> Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        # Add the positional embedding (original behavior).

        x = (x + positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class AudioEncoderTRT(nn.Module):
    """
    Audio Encoder using the TRT optimized engine.
    (This preserves the original method of slicing the positional embedding.)
    """

    def __init__(
        self, engine: torch2trt.TRTModule, positional_embedding: torch.Tensor
    ) -> None:
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # Compute n_audio_ctx as in the original implementation.

        n_audio_ctx = int(x.shape[2] // 2)
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        x = self.engine(x, pos_embed)
        return x


# -----------------------------------------------------------------------------
# TEXT DECODER
# -----------------------------------------------------------------------------


class _TextDecoderEngine(nn.Module):
    """
    Text Decoder Engine for Whisper TRT.
    """

    def __init__(self, blocks: nn.Module) -> None:
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, xa, mask)
        return x


class TextDecoderTRT(nn.Module):
    """
    Text Decoder using the TRT optimized engine.
    """

    def __init__(
        self,
        engine: torch2trt.TRTModule,
        token_embedding: nn.Embedding,
        positional_embedding: nn.Parameter,
        ln: LayerNorm,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        offset = 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        x = self.engine(x, xa, self.mask)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits


# -----------------------------------------------------------------------------
# WHISPER TRT MODEL
# -----------------------------------------------------------------------------


class WhisperTRT(nn.Module):
    """
    Whisper model optimized with TensorRT.

    This implementation preserves the original transcription behavior.
    It supports multiple languages via the language parameter in transcribe().
    """

    def __init__(
        self,
        dims: ModelDimensions,
        encoder: AudioEncoderTRT,
        decoder: TextDecoderTRT,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def embed_audio(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: Tensor) -> Tensor:
        return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: torch.Tensor) -> Tensor:
        return self.decoder(tokens, self.encoder(mel))

    @torch.no_grad()
    def transcribe(
        self, audio: str | np.ndarray, language: str = "auto"
    ) -> Dict[str, str]:
        """
        Transcribe audio input to text.

        If language is not "auto", sets the tokenizer's language accordingly.
        """
        if isinstance(audio, str):
            audio = whisper.audio.load_audio(audio)
        mel = whisper.audio.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)[
            None, ...
        ].cuda()
        if int(mel.shape[2]) > whisper.audio.N_FRAMES:
            mel = mel[:, :, : whisper.audio.N_FRAMES]
        audio_features = self.embed_audio(mel)

        # Configure tokenizer language.

        if self.tokenizer is not None:
            if language.lower() != "auto":
                lang_code = language.lower()
                if lang_code in TO_LANGUAGE_CODE:
                    lang_code = TO_LANGUAGE_CODE[lang_code]
                self.tokenizer.language = lang_code
                logger.debug(f"Tokenizer language set to: {lang_code}")
            else:
                self.tokenizer.language = None
                logger.debug("Tokenizer set to auto language detection.")
        else:
            logger.warning("No tokenizer found; transcription may be degraded.")
        # Greedy decoding loop (unchanged from the original).

        tokens = torch.LongTensor([self.tokenizer.sot]).cuda()[None, ...]
        for i in range(self.dims.n_text_ctx):
            logits = self.logits(tokens, audio_features)
            next_tokens = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_tokens[:, -1:]], dim=-1)
            if tokens[0, -1] == self.tokenizer.eot:
                break
        # Remove special tokens.

        tokens = tokens[:, 2:]
        tokens = tokens[:, :-1]
        text = self.tokenizer.decode(list(tokens.flatten().cpu().numpy()))
        # Remove internal control tokens using regex.

        text = re.sub(r"<\|transcribe\|><\|notimestamps\|>", "", text).strip()
        return {"text": text}


# -----------------------------------------------------------------------------
# BUILDER CLASSES FOR MULTILINGUAL AND ENGLISH-ONLY MODELS
# -----------------------------------------------------------------------------


class WhisperTRTBuilder:
    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 1 << 30
    verbose: bool = False
    _tokenizer: Optional[Tokenizer] = None  # Cache tokenizer instance.

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> torch2trt.TRTModule:
        model_inst = load_model(cls.model).cuda().eval()
        dims = model_inst.dims
        decoder_blocks_module = _TextDecoderEngine(model_inst.decoder.blocks)
        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()
        engine = torch2trt.torch2trt(
            decoder_blocks_module,
            [x, xa, mask],
            use_onnx=True,
            min_shapes=[
                (1, 1, dims.n_text_state),
                (1, 1, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
            opt_shapes=[
                (1, 1, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
            max_shapes=[
                (1, dims.n_text_ctx, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
            input_names=["x", "xa", "mask"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> torch2trt.TRTModule:
        model_inst = load_model(cls.model).cuda().eval()
        dims = model_inst.dims
        encoder_module = _AudioEncoderEngine(
            model_inst.encoder.conv1,
            model_inst.encoder.conv2,
            model_inst.encoder.blocks,
            model_inst.encoder.ln_post,
        )
        n_frames = dims.n_audio_ctx * 2
        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = model_inst.encoder.positional_embedding.cuda().detach()
        engine = torch2trt.torch2trt(
            encoder_module,
            [x, positional_embedding],
            use_onnx=True,
            min_shapes=[(1, dims.n_mels, 1), (1, dims.n_audio_state)],
            opt_shapes=[
                (1, dims.n_mels, n_frames),
                (dims.n_audio_ctx, dims.n_audio_state),
            ],
            max_shapes=[
                (1, dims.n_mels, n_frames),
                (dims.n_audio_ctx, dims.n_audio_state),
            ],
            input_names=["x", "positional_embedding"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls) -> Dict[str, Any]:
        model_inst = load_model(cls.model).cuda().eval()
        extra_state = {
            "token_embedding": model_inst.decoder.token_embedding.state_dict(),
            "positional_embedding": model_inst.decoder.positional_embedding,
            "ln": model_inst.decoder.ln.state_dict(),
            "mask": model_inst.decoder.mask,
        }
        return extra_state

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls) -> Dict[str, Any]:
        model_inst = load_model(cls.model).cuda().eval()
        extra_state = {"positional_embedding": model_inst.encoder.positional_embedding}
        return extra_state

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False) -> None:
        cls.verbose = verbose
        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": asdict(load_model(cls.model).dims),
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(),
        }
        torch.save(checkpoint, output_path)

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        # Cache the tokenizer instance so we do not reload every time.

        if cls._tokenizer is None:
            model_inst = load_model(cls.model)
            cls._tokenizer = whisper.tokenizer.get_tokenizer(
                model_inst.is_multilingual,
                num_languages=model_inst.num_languages,
                language=None,
                task="transcribe",
            )
        return cls._tokenizer

    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str) -> WhisperTRT:
        checkpoint = torch.load(trt_model_path)
        dims = ModelDimensions(**checkpoint["dims"])
        # Audio encoder.

        audio_encoder_engine = torch2trt.TRTModule()
        audio_encoder_engine.load_state_dict(checkpoint["audio_encoder_engine"])
        aes = checkpoint["audio_encoder_extra_state"]
        audio_positional_embedding = aes["positional_embedding"]
        encoder = AudioEncoderTRT(audio_encoder_engine, audio_positional_embedding)
        # Text decoder.

        text_decoder_engine = torch2trt.TRTModule()
        text_decoder_engine.load_state_dict(checkpoint["text_decoder_engine"])
        tes = checkpoint["text_decoder_extra_state"]
        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(tes["token_embedding"])
        text_positional_embedding = nn.Parameter(tes["positional_embedding"])
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(tes["ln"])
        text_mask = tes["mask"]
        decoder = TextDecoderTRT(
            text_decoder_engine,
            text_token_embedding,
            text_positional_embedding,
            text_ln,
            text_mask,
        )
        whisper_trt = WhisperTRT(dims, encoder, decoder, cls.get_tokenizer())
        whisper_trt = whisper_trt.cuda().eval()
        return whisper_trt


# -----------------------------------------------------------------------------
# ENGLISH-ONLY MODEL BUILDERS
# -----------------------------------------------------------------------------


class EnBuilder(WhisperTRTBuilder):
    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=False, num_languages=99, language="en", task="transcribe"
        )
        return tokenizer


class TinyEnBuilder(EnBuilder):
    model: str = "tiny.en"


class BaseEnBuilder(EnBuilder):
    model: str = "base.en"


class SmallEnBuilder(EnBuilder):
    model: str = "small.en"


# -----------------------------------------------------------------------------
# MULTILINGUAL MODEL BUILDERS
# -----------------------------------------------------------------------------


class TinyBuilder(WhisperTRTBuilder):
    model: str = "tiny"


class BaseBuilder(WhisperTRTBuilder):
    model: str = "base"


class SmallBuilder(WhisperTRTBuilder):
    model: str = "small"


class MediumBuilder(WhisperTRTBuilder):
    model: str = "medium"


class LargeBuilder(WhisperTRTBuilder):
    model: str = "large"


class LargeV2Builder(WhisperTRTBuilder):
    model: str = "large-v2"


class LargeV3Builder(WhisperTRTBuilder):
    model: str = "large-v3"


class LargeV3TurboBuilder(WhisperTRTBuilder):
    model: str = "large-v3-turbo"


# -----------------------------------------------------------------------------
# MODEL FILE-NAMING & BUILDER DICTIONARIES
# -----------------------------------------------------------------------------


MODEL_FILENAMES = {
    # English-only models:
    "tiny.en": "tiny_en_trt.pth",
    "base.en": "base_en_trt.pth",
    "small.en": "small_en_trt.pth",
    # Multilingual models:
    "tiny": "tiny_trt.pth",
    "base": "base_trt.pth",
    "small": "small_trt.pth",
    "medium": "medium_trt.pth",
    "large": "large_trt.pth",
    "large-v2": "large_v2_trt.pth",
    "large-v3": "large_v3_trt.pth",
    "large-v3-turbo": "large_v3_turbo_trt.pth",
}

MODEL_BUILDERS = {
    # English-only models:
    "tiny.en": TinyEnBuilder,
    "base.en": BaseEnBuilder,
    "small.en": SmallEnBuilder,
    # Multilingual models:
    "tiny": TinyBuilder,
    "base": BaseBuilder,
    "small": SmallBuilder,
    "medium": MediumBuilder,
    "large": LargeBuilder,
    "large-v2": LargeV2Builder,
    "large-v3": LargeV3Builder,
    "large-v3-turbo": LargeV3TurboBuilder,
}


def load_trt_model(
    name: str, path: Optional[str] = None, build: bool = True, verbose: bool = False
) -> WhisperTRT:
    if name not in MODEL_BUILDERS:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")
    if path is None:
        path = os.path.join(get_cache_dir(), MODEL_FILENAMES[name])
        make_cache_dir()
    builder = MODEL_BUILDERS[name]
    if not os.path.exists(path):
        if not build:
            raise RuntimeError(
                f"No model found at {path}. Please call load_trt_model with build=True."
            )
        else:
            builder.build(path, verbose=verbose)
    return builder.load(path)
