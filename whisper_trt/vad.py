import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import onnxruntime
import torch

from .cache import get_cache_dir, make_cache_dir
from .utils import check_file_md5, download_file

# Constants

SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx"
)
SILERO_VAD_ONNX_FILENAME = "silero_vad.onnx"
SILERO_VAD_ONNX_MD5_CHECKSUM = "03da8de2fec4108a089b39f1b4abefef"

SUPPORTED_LANGUAGES: List[str] = ["ru", "en", "de", "es"]

__all__ = ["SileroVAD", "load_vad"]

# Configure logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SileroVAD:
    """
    Silero Voice Activity Detection (VAD) using an ONNX model.
    """

    SAMPLE_RATES: List[int] = [8000, 16000]
    HIDDEN_SIZE: int = 64
    DEFAULT_BATCH_SIZE: int = 1

    def __init__(self, model_path: Path, force_cpu: bool = False) -> None:
        """
        Initialize the SileroVAD with the specified ONNX model.

        Args:
            model_path (Path): Path to the Silero VAD ONNX model file.
            force_cpu (bool): If True, forces the use of CPU for inference.
        """
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        providers = ["CPUExecutionProvider"] if force_cpu else None

        try:
            self.session = onnxruntime.InferenceSession(
                str(model_path), sess_options=session_options, providers=providers
            )
            logger.debug(
                f"Initialized ONNX Runtime session with providers: {providers or 'default'}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise RuntimeError(
                f"ONNX Runtime session initialization failed: {e}"
            ) from e
        self.reset_states(batch_size=self.DEFAULT_BATCH_SIZE)
        self.sample_rates = self.SAMPLE_RATES.copy()
        logger.debug(f"SileroVAD initialized with sample rates: {self.sample_rates}")

    def _validate_input(self, x: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        Validate and preprocess the input audio tensor.

        Args:
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the input audio.

        Returns:
            Tuple[torch.Tensor, int]: Preprocessed audio tensor and updated sample rate.

        Raises:
            ValueError: If input dimensions are incorrect or sample rate is unsupported.
        """
        logger.debug(f"Validating input with shape {x.shape} and sample rate {sr}")

        if x.dim() == 1:
            x = x.unsqueeze(0)
            logger.debug("Input tensor reshaped to add batch dimension.")
        elif x.dim() > 2:
            error_msg = f"Too many dimensions for input audio chunk: {x.dim()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000
            logger.debug(f"Resampled audio by step {step} to achieve 16000 Hz.")
        if sr not in self.sample_rates:
            error_msg = f"Unsupported sampling rate: {sr}. Supported rates: {self.sample_rates} or multiples of 16000."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Ensure the chunk is not too short

        min_length = 512  # Based on num_samples default in audio_forward
        if x.shape[1] < min_length:
            error_msg = "Input audio chunk is too short."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Input validated. Final shape: {x.shape}, Sample rate: {sr}")
        return x, sr

    def reset_states(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Reset the internal states of the VAD model.

        Args:
            batch_size (int): The batch size for the VAD model.
        """
        self._h = np.zeros((2, batch_size, self.HIDDEN_SIZE), dtype=np.float32)
        self._c = np.zeros((2, batch_size, self.HIDDEN_SIZE), dtype=np.float32)
        self._last_sr: Optional[int] = None
        self._last_batch_size: Optional[int] = None
        logger.debug(f"States reset for batch size {batch_size}.")

    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Perform VAD on the input audio tensor.

        Args:
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the input audio.

        Returns:
            torch.Tensor: VAD output tensor.

        Raises:
            ValueError: If sample rate is unsupported.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            logger.debug("Converted input from NumPy array to torch.Tensor.")
        x, sr = self._validate_input(x, sr)
        batch_size = x.size(0)
        logger.debug(f"Processing batch size {batch_size}.")

        # Reset states if sample rate or batch size has changed

        if (self._last_sr is not None and self._last_sr != sr) or (
            self._last_batch_size is not None and self._last_batch_size != batch_size
        ):
            logger.debug("Sample rate or batch size changed. Resetting states.")
            self.reset_states(batch_size=batch_size)
        self._last_sr = sr
        self._last_batch_size = batch_size

        ort_inputs = {
            "input": x.numpy(),
            "h": self._h,
            "c": self._c,
            "sr": np.array(sr, dtype=np.int64),
        }

        try:
            ort_outputs = self.session.run(None, ort_inputs)
            logger.debug("ONNX Runtime inference successful.")
        except Exception as e:
            logger.error(f"ONNX Runtime inference failed: {e}")
            raise RuntimeError(f"ONNX Runtime inference failed: {e}") from e
        out, self._h, self._c = ort_outputs
        logger.debug(
            f"Output shape: {out.shape}, Updated hidden states shape: {self._h.shape}, {self._c.shape}"
        )

        return torch.tensor(out)

    def audio_forward(
        self, x: torch.Tensor, sr: int, num_samples: int = 512
    ) -> torch.Tensor:
        """
        Perform VAD on audio by processing it in chunks.

        Args:
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the input audio.
            num_samples (int): Number of samples per chunk.

        Returns:
            torch.Tensor: Concatenated VAD outputs.

        Raises:
            ValueError: If audio is too short.
        """
        logger.debug(f"Starting audio_forward with num_samples={num_samples}")
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples != 0:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", 0.0)
            logger.debug(
                f"Padded input audio with {pad_num} zeros to make it divisible by {num_samples}."
            )
        self.reset_states(batch_size=x.size(0))
        logger.debug(f"Processing audio in chunks of {num_samples} samples.")

        for i in range(0, x.size(1), num_samples):
            wavs_batch = x[:, i : i + num_samples]
            logger.debug(
                f"Processing chunk {i // num_samples + 1}: samples {i} to {i + num_samples}"
            )
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)
        stacked = torch.cat(outs, dim=1).cpu()
        logger.debug(f"Concatenated output shape: {stacked.shape}")
        return stacked


def load_vad(
    model_path: Optional[str] = None, download: bool = True, force_cpu: bool = False
) -> SileroVAD:
    """
    Load the Silero VAD model.

    Args:
        model_path (Optional[str]): Path to the Silero VAD ONNX model. If None, it will be downloaded to the cache directory.
        download (bool): Whether to download the model if it's not found at the specified path.
        force_cpu (bool): If True, forces the use of CPU for inference.

    Returns:
        SileroVAD: An instance of the SileroVAD class.

    Raises:
        RuntimeError: If the model file is not found and downloading is disabled, or if the MD5 checksum does not match.
    """
    if model_path is None:
        make_cache_dir()
        model_path = Path(get_cache_dir()) / SILERO_VAD_ONNX_FILENAME
        logger.debug(f"No model path provided. Using cache path: {model_path}")
    else:
        model_path = Path(model_path)
        logger.debug(f"Using provided model path: {model_path}")
    if not model_path.exists():
        if not download:
            error_msg = f"VAD model not found at {model_path}. Please set download=True to download it."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(
            f"Model not found at {model_path}. Downloading from {SILERO_VAD_ONNX_URL}"
        )
        download_file(SILERO_VAD_ONNX_URL, str(model_path))
        logger.info(f"Model downloaded to {model_path}")
    if not check_file_md5(str(model_path), SILERO_VAD_ONNX_MD5_CHECKSUM):
        error_msg = (
            f"The MD5 checksum for {model_path} does not match the expected value."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.debug(f"MD5 checksum for {model_path} verified successfully.")

    return SileroVAD(model_path=model_path, force_cpu=force_cpu)
