import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch2trt
import tensorrt

from dataclasses import asdict

from whisper import load_model
from whisper.model import (
    LayerNorm,
    Linear,
    Tensor,
    ModelDimensions,
    Whisper,
)
from whisper.tokenizer import Tokenizer, LANGUAGES, TO_LANGUAGE_CODE
import whisper.audio

from .cache import get_cache_dir, make_cache_dir
from .__version__ import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _AudioEncoderEngine(nn.Module):
    """
    Audio Encoder Engine for Whisper TRT.

    This module allows for online substitution of positional embeddings.
    """

    def __init__(
        self,
        conv1: nn.Conv1d,
        conv2: nn.Conv1d,
        blocks: nn.ModuleList,
        ln_post: LayerNorm,
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.blocks = blocks
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x: Tensor, positional_embedding: Tensor) -> Tensor:
        """
        Forward pass for the Audio Encoder.

        Args:
            x (Tensor): Input tensor.
            positional_embedding (Tensor): Positional embeddings.

        Returns:
            Tensor: Output tensor after encoding.
        """
        logger.debug("AudioEncoderEngine forward pass started.")
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + positional_embedding).to(x.dtype)
        logger.debug("Positional embedding added.")

        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        logger.debug("LayerNorm applied.")

        return x


class AudioEncoderTRT(nn.Module):
    """
    Audio Encoder using TensorRT optimized engine.
    """

    def __init__(
        self, engine: torch2trt.TRTModule, positional_embedding: torch.Tensor
    ) -> None:
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Audio Encoder TRT.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Encoded audio features.
        """
        n_audio_ctx = x.shape[2] // 2
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        logger.debug(f"Using positional embedding with context size: {n_audio_ctx}")
        return self.engine(x, pos_embed)


class _TextDecoderEngine(nn.Module):
    """
    Text Decoder Engine for Whisper TRT.
    """

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for the Text Decoder.

        Args:
            x (Tensor): Input tensor.
            xa (Tensor): Audio features tensor.
            mask (Tensor): Attention mask tensor.

        Returns:
            Tensor: Output tensor after decoding.
        """
        logger.debug("TextDecoderEngine forward pass started.")
        for block in self.blocks:
            x = block(x, xa, mask)
        return x


class TextDecoderTRT(nn.Module):
    """
    Text Decoder using TensorRT optimized engine.
    """

    def __init__(
        self,
        engine: torch2trt.TRTModule,
        token_embedding: nn.Embedding,
        positional_embedding: nn.Parameter,
        ln: LayerNorm,
        mask: Tensor,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        """
        Forward pass for the Text Decoder TRT.

        Args:
            x (Tensor): Input tokens tensor.
            xa (Tensor): Audio features tensor.

        Returns:
            Tensor: Logits tensor.
        """
        logger.debug("TextDecoderTRT forward pass started.")
        offset = 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        logger.debug("Token and positional embeddings added.")

        x = self.engine(x, xa, self.mask)
        logger.debug("Engine processed the input.")

        x = self.ln(x)
        logger.debug("LayerNorm applied.")

        logits = (x @ self.token_embedding.weight.to(x.dtype).transpose(0, 1)).float()
        logger.debug("Logits computed.")

        return logits


class WhisperTRT(nn.Module):
    """
    Whisper model optimized with TensorRT.
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
        self.stream = torch.cuda.Stream()  # Create a CUDA stream

    def embed_audio(self, mel: Tensor) -> Tensor:
        with torch.cuda.stream(self.stream):
            logger.debug("Embedding audio features.")
            return self.encoder(mel)

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        with torch.cuda.stream(self.stream):
            logger.debug("Computing logits.")
            return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        with torch.cuda.stream(self.stream):
            logger.debug("WhisperTRT forward pass started.")
            audio_features = self.encoder(mel)
            return self.decoder(tokens, audio_features)

    @torch.no_grad()
    def transcribe(
        self, audio: str | np.ndarray, language: str = "auto"
    ) -> Dict[str, str]:
        """
        Transcribe audio input to text, supporting multilingual or auto-detection.
        """
        logger.debug("Transcription started.")

        # 1. Handle language selection in the tokenizer
        if self.tokenizer is not None:
            if language.lower() != "auto":
                lang_code = language.lower()
                if lang_code not in LANGUAGES and lang_code not in TO_LANGUAGE_CODE:
                    raise ValueError(f"Unsupported language code '{language}'.")
                if lang_code in TO_LANGUAGE_CODE:
                    lang_code = TO_LANGUAGE_CODE[lang_code]
                self.tokenizer.language = lang_code
                logger.debug(f"Tokenizer language set to: {lang_code}")
            else:
                self.tokenizer.language = None
                logger.debug("Tokenizer set to auto language detection.")
        else:
            logger.warning(
                "No tokenizer found; transcription might fail or be monolingual."
            )

        # 2. Load/process audio
        if isinstance(audio, str):
            audio = whisper.audio.load_audio(audio)
        mel = whisper.audio.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)[
            None, ...
        ].cuda()

        if mel.shape[2] > whisper.audio.N_FRAMES:
            mel = mel[:, :, : whisper.audio.N_FRAMES]
            logger.debug("Truncated mel spectrogram to fit N_FRAMES.")

        # 3. Encode
        audio_features = self.embed_audio(mel)

        # 4. Decode token-by-token
        tokens = torch.LongTensor([[self.tokenizer.sot]]).cuda()
        for i in range(self.dims.n_text_ctx):
            logits = self.logits(tokens, audio_features)
            next_tokens = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_tokens[:, -1:]], dim=-1)
            if tokens[0, -1] == self.tokenizer.eot:
                logger.debug(f"End of transcription detected at step {i}.")
                break

        # 5. Manually remove special tokens from the final text
        tokens = tokens[:, 2:-1]
        text = self.tokenizer.decode(tokens.flatten().tolist())

        # Remove placeholders like <|...|> using regex
        text = re.sub(r"<\|.*?\|>", "", text)
        logger.debug(f"Final decoded text after regex filter: {text}")

        return {"text": text}

    @torch.no_grad()
    def get_supported_languages(self) -> List[str]:
        """
        Returns a list of supported language codes for this model.
        If the tokenizer is missing or the model is monolingual, default to ['en'].
        """
        if self.tokenizer is None:
            logger.debug("No tokenizer available. Falling back to ['en'].")
            return ["en"]
        try:
            codes = list(self.tokenizer.all_language_codes)
            return codes if codes else ["en"]
        except AttributeError:
            logger.debug("Tokenizer missing 'all_language_codes'. Returning ['en'].")
            return ["en"]


class WhisperTRTBuilder:
    """
    Builder class for constructing WhisperTRT models with TensorRT optimizations.
    """

    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 1 << 30  # 1 GB
    verbose: bool = False

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> torch2trt.TRTModule:
        logger.debug(f"Building text decoder engine for model '{cls.model}'.")
        model = load_model(cls.model).cuda().eval()
        dims = model.dims

        decoder_module = _TextDecoderEngine(model.decoder.blocks)

        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()

        engine = torch2trt.torch2trt(
            decoder_module,
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
        logger.debug("Text decoder engine built successfully.")
        return engine

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> torch2trt.TRTModule:
        logger.debug(f"Building audio encoder engine for model '{cls.model}'.")
        model = load_model(cls.model).cuda().eval()
        dims = model.dims

        encoder_module = _AudioEncoderEngine(
            conv1=model.encoder.conv1,
            conv2=model.encoder.conv2,
            blocks=model.encoder.blocks,
            ln_post=model.encoder.ln_post,
        )

        n_frames = dims.n_audio_ctx * 2
        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = model.encoder.positional_embedding.cuda().detach()

        engine = torch2trt.torch2trt(
            encoder_module,
            [x, positional_embedding],
            use_onnx=True,
            min_shapes=[
                (1, dims.n_mels, 1),
                (1, dims.n_audio_state),
            ],
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
        logger.debug("Audio encoder engine built successfully.")
        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls) -> Dict[str, Any]:
        logger.debug(f"Retrieving text decoder extra state for model '{cls.model}'.")
        model = load_model(cls.model).cuda().eval()

        extra_state = {
            "token_embedding": model.decoder.token_embedding.state_dict(),
            "positional_embedding": model.decoder.positional_embedding,
            "ln": model.decoder.ln.state_dict(),
            "mask": model.decoder.mask,
        }
        logger.debug("Text decoder extra state retrieved.")
        return extra_state

    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls) -> Dict[str, Any]:
        logger.debug(f"Retrieving audio encoder extra state for model '{cls.model}'.")
        model = load_model(cls.model).cuda().eval()

        extra_state = {
            "positional_embedding": model.encoder.positional_embedding,
        }
        logger.debug("Audio encoder extra state retrieved.")
        return extra_state

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False) -> None:
        cls.verbose = verbose
        logger.debug(f"Building WhisperTRT model and saving to '{output_path}'.")

        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": asdict(load_model(cls.model).dims),
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)
        logger.info(f"WhisperTRT model saved successfully at '{output_path}'.")

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        model = load_model(cls.model)
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=model.is_multilingual,
            num_languages=model.num_languages,
            language=None,  # Not forcing 'en'
            task="transcribe",
        )
        logger.debug("Tokenizer retrieved with language=None for multilingual usage.")
        return tokenizer

    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str) -> "WhisperTRT":
        logger.debug(f"Loading WhisperTRT model from '{trt_model_path}'.")
        checkpoint = torch.load(trt_model_path, map_location="cuda")

        dims = ModelDimensions(**checkpoint["dims"])

        audio_encoder_engine = torch2trt.TRTModule()
        audio_encoder_engine.load_state_dict(checkpoint["audio_encoder_engine"])
        audio_encoder_extra_state = checkpoint["audio_encoder_extra_state"]
        audio_positional_embedding = audio_encoder_extra_state["positional_embedding"]

        encoder = AudioEncoderTRT(
            engine=audio_encoder_engine,
            positional_embedding=audio_positional_embedding,
        )
        logger.debug("Audio encoder loaded.")

        text_decoder_engine = torch2trt.TRTModule()
        text_decoder_engine.load_state_dict(checkpoint["text_decoder_engine"])
        text_decoder_extra_state = checkpoint["text_decoder_extra_state"]

        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(
            text_decoder_extra_state["token_embedding"]
        )

        text_positional_embedding = nn.Parameter(
            text_decoder_extra_state["positional_embedding"]
        )
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(text_decoder_extra_state["ln"])

        text_mask = text_decoder_extra_state["mask"]

        decoder = TextDecoderTRT(
            engine=text_decoder_engine,
            token_embedding=text_token_embedding,
            positional_embedding=text_positional_embedding,
            ln=text_ln,
            mask=text_mask,
        )
        logger.debug("Text decoder loaded.")

        tokenizer = cls.get_tokenizer()

        whisper_trt = WhisperTRT(
            dims=dims, encoder=encoder, decoder=decoder, tokenizer=tokenizer
        )
        whisper_trt = whisper_trt.cuda().eval()
        logger.info("WhisperTRT model loaded successfully.")

        return whisper_trt


class EnBuilder(WhisperTRTBuilder):
    """
    English language model builder.
    """

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        """
        English-only builder: still forces language='en'.
        """
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=False,  # English-only model
            num_languages=99,
            language="en",
            task="transcribe",
        )
        logger.debug("English tokenizer retrieved for EnBuilder.")
        return tokenizer


class TinyEnBuilder(EnBuilder):
    """Builder for 'tiny.en'"""

    model: str = "tiny.en"


class BaseEnBuilder(EnBuilder):
    """Builder for 'base.en'"""

    model: str = "base.en"


class SmallEnBuilder(EnBuilder):
    """Builder for 'small.en'"""

    model: str = "small.en"


class MediumEnBuilder(EnBuilder):
    """Builder for 'medium.en'"""

    model: str = "medium.en"


class TinyBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'tiny'"""

    model: str = "tiny"


class BaseBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'base'"""

    model: str = "base"


class SmallBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'small'"""

    model: str = "small"


class MediumBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'medium'"""

    model: str = "medium"


class LargeBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'large'"""

    model: str = "large"


class LargeV2Builder(WhisperTRTBuilder):
    """Builder for multilingual 'large-v2'"""

    model: str = "large-v2"


class LargeV3Builder(WhisperTRTBuilder):
    """Builder for multilingual 'large-v3'"""

    model: str = "large-v3"


class LargeV3TurboBuilder(WhisperTRTBuilder):
    """Builder for multilingual 'large-v3-turbo'"""

    model: str = "large-v3-turbo"


MODEL_FILENAMES: Dict[str, str] = {
    "tiny.en": "tiny_en_trt.pth",
    "base.en": "base_en_trt.pth",
    "small.en": "small_en_trt.pth",
    "medium.en": "medium_en_trt.pth",
    "tiny": "tiny_trt.pth",
    "base": "base_trt.pth",
    "small": "small_trt.pth",
    "medium": "medium_trt.pth",
    "large": "large_trt.pth",
    "large-v2": "large_v2_trt.pth",
    "large-v3": "large_v3_trt.pth",
    "large-v3-turbo": "large_v3_turbo_trt.pth",
}

MODEL_BUILDERS: Dict[str, Any] = {
    "tiny.en": TinyEnBuilder,
    "base.en": BaseEnBuilder,
    "small.en": SmallEnBuilder,
    "medium.en": MediumEnBuilder,
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
    name: str,
    path: Optional[str] = None,
    build: bool = True,
    verbose: bool = False,
) -> WhisperTRT:
    """
    Load a TensorRT optimized Whisper model.

    Args:
        name (str): Name of the model (e.g., 'tiny.en', 'base.en', 'small.en', 'base', 'small').
        path (Optional[str], optional): Path to the model checkpoint.
                                        If None, uses the cache directory. Defaults to None.
        build (bool, optional): If True, builds the model if not found. Defaults to True.
        verbose (bool, optional): If True, enables verbose logging. Defaults to False.

    Returns:
        WhisperTRT: The loaded WhisperTRT model.

    Raises:
        RuntimeError: If the model name is unsupported or if building/loading fails.
    """
    logger.debug(
        f"Loading TensorRT model '{name}' with build={build} and verbose={verbose}."
    )

    logger.debug(f"Using torch version: {torch.__version__}")
    logger.debug(f"Using TensorRT version: {tensorrt.__version__}")

    if name not in MODEL_BUILDERS:
        error_msg = f"Model '{name}' is not supported by WhisperTRT."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    builder_cls = MODEL_BUILDERS[name]

    if path is None:
        make_cache_dir()
        cache_dir = get_cache_dir()
        path = str(Path(cache_dir) / MODEL_FILENAMES[name])
        logger.debug(f"No path provided. Using cache path: {path}")
    else:
        path = str(Path(path))
        logger.debug(f"Using provided path: {path}")
    if not Path(path).exists():
        if not build:
            error_msg = (
                f"No model found at {path}. Please set build=True to build the model."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"Model not found at {path}. Building the model.")
        builder_cls.build(path, verbose=verbose)
    try:
        model = builder_cls.load(path)
        logger.info(f"Model '{name}' loaded successfully from '{path}'.")
        return model
    except Exception as e:
        error_msg = f"Failed to load model '{name}' from '{path}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
