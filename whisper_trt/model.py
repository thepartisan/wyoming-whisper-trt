# SPDX-License-Identifier: MIT

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch2trt
import tensorrt
import numpy as np
import whisper
from whisper import load_model
from whisper.model import LayerNorm, Tensor, ModelDimensions
from whisper.tokenizer import Tokenizer
from typing import Optional, Dict, List, Union
from dataclasses import asdict
import logging

from .cache import get_cache_dir, make_cache_dir
from .__version__ import __version__

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class _AudioEncoderEngine(nn.Module):
    """
    Audio Encoder module to be converted to TensorRT engine.
    Allows for online substitution of positional embedding.
    """
    def __init__(self, conv1, conv2, blocks, ln_post):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.blocks = blocks
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x: Tensor, positional_embedding: Tensor) -> Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch_size, n_frames, n_mels)
        x = (x + positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x

class AudioEncoderTRT(nn.Module):
    """
    TensorRT-optimized Audio Encoder.
    """
    def __init__(
        self,
        engine: torch2trt.TRTModule,
        positional_embedding: torch.Tensor,
    ):
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        n_audio_ctx = int(x.shape[2] // 2)
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        x = x.to(self.positional_embedding.device, self.positional_embedding.dtype)
        pos_embed = pos_embed.to(x.device, x.dtype)
        output = self.engine(x, pos_embed)
        return output

class _TextDecoderEngine(nn.Module):
    """
    Text Decoder module to be converted to TensorRT engine.
    """
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, mask, cross_attn=xa)  # Pass xa as a keyword argument
        return x

class TextDecoderTRT(nn.Module):
    """
    TensorRT-optimized Text Decoder.
    """
    def __init__(
        self,
        engine: torch2trt.TRTModule,
        token_embedding: nn.Embedding,
        positional_embedding: nn.Parameter,
        ln: LayerNorm,
        mask: torch.Tensor,
    ):
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        offset = 0  # For generality; modify if needed
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        x = self.engine(x, xa, self.mask)  # Correct argument order: x, xa, mask
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits

class WhisperTRT(nn.Module):
    """
    A TensorRT-optimized version of the Whisper model for efficient inference.

    Attributes:
        dims (ModelDimensions): Dimensions of the Whisper model.
        encoder (AudioEncoderTRT): The audio encoder module optimized with TensorRT.
        decoder (TextDecoderTRT): The text decoder module optimized with TensorRT.
        tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
        stream (torch.cuda.Stream): CUDA stream for asynchronous execution.
    """
    def __init__(
        self,
        dims: ModelDimensions,
        encoder: AudioEncoderTRT,
        decoder: TextDecoderTRT,
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.stream = torch.cuda.Stream()

    @torch.no_grad()
    def embed_audio(self, mel: Tensor) -> Tensor:
        with torch.cuda.stream(self.stream):
            output = self.encoder(mel)
        torch.cuda.current_stream().wait_stream(self.stream)
        return output

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> Tensor:
        with torch.cuda.stream(self.stream):
            output = self.decoder(tokens, audio_features)
        torch.cuda.current_stream().wait_stream(self.stream)
        return output

    @torch.no_grad()
    def transcribe(self, audio: Union[str, np.ndarray]) -> Dict[str, str]:
        """
        Transcribes the given audio file or numpy array.

        Args:
            audio (str | np.ndarray): Path to the audio file or numpy array of audio data.

        Returns:
            Dict[str, str]: A dictionary containing the transcribed text.
        """
        # Load and preprocess audio
        if isinstance(audio, str):
            if not os.path.isfile(audio):
                raise FileNotFoundError(f"Audio file '{audio}' not found.")
            audio = whisper.audio.load_audio(audio)
        elif isinstance(audio, np.ndarray):
            if audio.ndim > 2:
                raise ValueError("Audio input must be a 1D or 2D numpy array.")
        else:
            raise TypeError("Audio input must be a file path or a numpy array.")

        # Ensure audio is correctly sampled
        if audio.shape[0] != whisper.audio.SAMPLE_RATE:
            audio = whisper.audio.resample_audio(audio, whisper.audio.SAMPLE_RATE)

        mel = whisper.audio.log_mel_spectrogram(
            audio, padding=whisper.audio.N_SAMPLES
        )[None, ...].cuda()

        if mel.shape[2] > whisper.audio.N_FRAMES:
            mel = mel[:, :, : whisper.audio.N_FRAMES]

        audio_features = self.embed_audio(mel)

        tokens = torch.full(
            (1, self.dims.n_text_ctx),
            fill_value=self.tokenizer.eot,
            dtype=torch.long,
            device='cuda',
        )
        tokens[0, 0] = self.tokenizer.sot

        for i in range(1, self.dims.n_text_ctx):
            logits = self.logits(tokens[:, :i], audio_features)  # Now correctly passing two arguments
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens[0, i] = next_token
            if next_token == self.tokenizer.eot:
                break

        tokens = tokens[:, 1 : i + 1]
        text = self.tokenizer.decode(tokens[0].tolist())
        result = {"text": text}
        return result

class WhisperTRTBuilder:
    """
    Builder class for creating and loading TensorRT-optimized Whisper models.
    """
    supported_models = ["tiny.en", "base.en", "small.en", "medium.en", "large.en"]
    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 1 << 30
    verbose: bool = False

    @classmethod
    def set_model(cls, model_name: str):
        if model_name not in cls.supported_models:
            raise ValueError(f"Model '{model_name}' is not supported.")
        cls.model = model_name

    @classmethod
    @torch.no_grad()
    def build_engine(
        cls,
        module: nn.Module,
        inputs: List[Tensor],
        shapes: Dict[str, List],
        input_names: List[str],
        output_names: List[str],
    ) -> torch2trt.TRTModule:
        logger.info("Building TensorRT engine...")
        engine = torch2trt.torch2trt(
            module,
            inputs,
            use_onnx=True,
            min_shapes=shapes['min'],
            opt_shapes=shapes['opt'],
            max_shapes=shapes['max'],
            input_names=input_names,
            output_names=output_names,
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
            log_level=(
                tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR
            ),
        )
        logger.info("TensorRT engine built successfully.")
        return engine

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls, model) -> torch2trt.TRTModule:
        dims = model.dims

        decoder_blocks_module = _TextDecoderEngine(
            model.decoder.blocks
        )

        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = model.decoder.mask.cuda()

        shapes = {
            'min': [
                (1, 1, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
            'opt': [
                (1, dims.n_text_ctx, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
            'max': [
                (1, dims.n_text_ctx, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx),
            ],
        }

        engine = cls.build_engine(
            decoder_blocks_module,
            [x, xa, mask],
            shapes,
            input_names=["x", "xa", "mask"],  # Ensure the order matches the forward method
            output_names=["output"],
        )

        return engine

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls, model) -> torch2trt.TRTModule:
        dims = model.dims

        encoder_module = _AudioEncoderEngine(
            model.encoder.conv1,
            model.encoder.conv2,
            model.encoder.blocks,
            model.encoder.ln_post
        )

        n_frames = dims.n_audio_ctx * 2

        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = model.encoder.positional_embedding.cuda().detach()

        shapes = {
            'min': [
                (1, dims.n_mels, 1),
                (1, dims.n_audio_state),
            ],
            'opt': [
                (1, dims.n_mels, n_frames),
                (dims.n_audio_ctx, dims.n_audio_state),
            ],
            'max': [
                (1, dims.n_mels, n_frames),
                (dims.n_audio_ctx, dims.n_audio_state),
            ],
        }

        engine = cls.build_engine(
            encoder_module,
            [x, positional_embedding],
            shapes,
            input_names=["x", "positional_embedding"],
            output_names=["output"],
        )

        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls, model):
        extra_state = {
            "token_embedding": model.decoder.token_embedding.state_dict(),
            "positional_embedding": model.decoder.positional_embedding,
            "ln": model.decoder.ln.state_dict(),
            "mask": model.decoder.mask,
        }
        return extra_state

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls, model):
        extra_state = {
            "positional_embedding": model.encoder.positional_embedding
        }
        return extra_state

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False):
        cls.verbose = verbose
        logger.info(f"Building Whisper TensorRT model: {cls.model}")

        model = cls.load_model()

        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": asdict(model.dims),
            "text_decoder_engine": cls.build_text_decoder_engine(model).state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(model),
            "audio_encoder_engine": cls.build_audio_encoder_engine(model).state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(model),
        }

        torch.save(checkpoint, output_path)
        logger.info(f"Model saved at {output_path}")

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        if not hasattr(cls, '_tokenizer'):
            model = cls.load_model()
            cls._tokenizer = whisper.tokenizer.get_tokenizer(
                model.is_multilingual,
                num_languages=model.num_languages,
                language="en",
                task="transcribe",
            )
        return cls._tokenizer

    @classmethod
    @torch.no_grad()
    def load_model(cls):
        try:
            model = load_model(cls.model).cuda().eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{cls.model}': {e}")
            raise

    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str):
        checkpoint = torch.load(trt_model_path)
        dims = ModelDimensions(**checkpoint['dims'])

        # Audio Encoder
        audio_encoder_engine = torch2trt.TRTModule()
        audio_encoder_engine.load_state_dict(checkpoint['audio_encoder_engine'])
        aes = checkpoint['audio_encoder_extra_state']
        audio_positional_embedding = aes['positional_embedding']
        encoder = AudioEncoderTRT(
            audio_encoder_engine,
            audio_positional_embedding,
        )

        # Text Decoder
        text_decoder_engine = torch2trt.TRTModule()
        text_decoder_engine.load_state_dict(checkpoint['text_decoder_engine'])
        tes = checkpoint['text_decoder_extra_state']
        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(tes['token_embedding'])
        text_positional_embedding = nn.Parameter(tes['positional_embedding'])
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(tes['ln'])
        text_mask = tes['mask']

        decoder = TextDecoderTRT(
            text_decoder_engine, 
            text_token_embedding,
            text_positional_embedding, 
            text_ln, 
            text_mask
        )

        whisper_trt = WhisperTRT(dims, encoder, decoder, cls.get_tokenizer())
        whisper_trt = whisper_trt.cuda().eval()

        return whisper_trt

# Additional Builders for specific models
class EnBuilder(WhisperTRTBuilder):
    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        if not hasattr(cls, '_tokenizer'):
            cls._tokenizer = whisper.tokenizer.get_tokenizer(
                False,
                num_languages=99,
                language="en",
                task="transcribe",
            )
        return cls._tokenizer

class TinyEnBuilder(EnBuilder):
    model: str = "tiny.en"

class BaseEnBuilder(EnBuilder):
    model: str = "base.en"

class SmallEnBuilder(EnBuilder):
    model: str = "small.en"

MODEL_FILENAMES = {
    "tiny.en": "tiny_en_trt.pth",
    "base.en": "base_en_trt.pth",
    "small.en": "small_en_trt.pth",
    # Add other models if needed
}

MODEL_BUILDERS = {
    "tiny.en": TinyEnBuilder,
    "base.en": BaseEnBuilder,
    "small.en": SmallEnBuilder,
    # Add other models if needed
}

def load_trt_model(
    name: str,
    path: Optional[str] = None,
    build: bool = True,
    verbose: bool = False,
) -> WhisperTRT:
    """
    Loads or builds a TensorRT-optimized Whisper model.

    Args:
        name (str): Name of the model.
        path (Optional[str]): Path to the model file.
        build (bool): Whether to build the model if not found.
        verbose (bool): Verbose output during building.

    Returns:
        WhisperTRT: The loaded or built WhisperTRT model.
    """
    if name not in MODEL_BUILDERS:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")

    if path is None:
        make_cache_dir()
        path = os.path.join(get_cache_dir(), MODEL_FILENAMES[name])

    builder_class = MODEL_BUILDERS[name]
    builder_class.set_model(name)

    if not os.path.exists(path):
        if not build:
            raise RuntimeError(f"No model found at {path}. Please call load_trt_model with build=True.")
        else:
            builder_class.build(path, verbose=verbose)

    return builder_class.load(path)
