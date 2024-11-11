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

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

from .cache import get_cache_dir, make_cache_dir
from .utils import check_file_md5, download_file

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Constants
SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx"
)
SILERO_VAD_ONNX_FILENAME = "silero_vad.onnx"
SILERO_VAD_ONNX_MD5_CHECKSUM = "03da8de2fec4108a089b39f1b4abefef"

SUPPORTED_LANGUAGES = ["ru", "en", "de", "es"]
SUPPORTED_SAMPLE_RATES = [8000, 16000]
MAX_INPUT_RATIO = 31.25  # sr / x.shape[1] > 31.25


__all__ = ["SileroVAD", "load_vad"]


class SileroVAD:
    """
    Silero Voice Activity Detection (VAD) using an ONNX model for efficient inference.

    Attributes:
        session (onnxruntime.InferenceSession): The ONNX Runtime session for inference.
        sample_rates (List[int]): Supported audio sample rates.
        _h (np.ndarray): Hidden state for the VAD model.
        _c (np.ndarray): Cell state for the VAD model.
        _last_sr (int): Last sample rate used.
        _last_batch_size (int): Last batch size used.
    """

    def __init__(self, path: Union[str, Path], force_onnx_cpu: bool = False) -> None:
        """
        Initializes the SileroVAD instance by loading the ONNX model.

        Args:
            path (Union[str, Path]): Path to the ONNX model file.
            force_onnx_cpu (bool, optional): Whether to force the use of CPU execution provider. Defaults to False.

        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
            RuntimeError: If the ONNX Runtime session cannot be created.
        """
        model_path = Path(path)
        if not model_path.is_file():
            logger.error(f"VAD model file not found at '{model_path}'.")
            raise FileNotFoundError(f"VAD model file not found at '{model_path}'.")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        providers = ["CPUExecutionProvider"] if force_onnx_cpu else None
        try:
            self.session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=providers,
            )
            logger.info(f"Loaded Silero VAD model from '{model_path}'.")
        except onnxruntime.OrtException as e:
            logger.error(f"Failed to create ONNX Runtime session: {e}")
            raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")

        self.sample_rates = SUPPORTED_SAMPLE_RATES
        self.reset_states()
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        Validates and preprocesses the input audio tensor.

        Args:
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the audio.

        Returns:
            Tuple[torch.Tensor, int]: Validated and possibly resampled audio tensor and updated sample rate.

        Raises:
            ValueError: If input dimensions are invalid, sample rate is unsupported, or input audio is too short.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            logger.debug("Input audio was 1D; added batch dimension.")

        if x.dim() > 2:
            logger.error(f"Too many dimensions for input audio chunk: {x.dim()}.")
            raise ValueError(f"Too many dimensions for input audio chunk: {x.dim()}.")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000
            logger.debug(f"Resampled audio from {sr * step} Hz to {sr} Hz.")

        if sr not in self.sample_rates:
            logger.error(
                f"Unsupported sample rate: {sr}. Supported rates: {self.sample_rates}."
            )
            raise ValueError(
                f"Supported sample rates: {self.sample_rates} (or multiples of 16000)."
            )

        if sr / x.shape[1] > MAX_INPUT_RATIO:
            logger.error("Input audio chunk is too short.")
            raise ValueError("Input audio chunk is too short.")

        logger.debug(f"Input audio validated with sample rate: {sr} Hz.")
        return x, sr

    def reset_states(self, batch_size: int = 1) -> None:
        """
        Resets the hidden and cell states of the VAD model.

        Args:
            batch_size (int, optional): Batch size for the states. Defaults to 1.
        """
        self._h = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._c = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0
        logger.debug(f"Reset VAD states with batch size: {batch_size}.")

    def __call__(self, x: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        """
        Performs VAD on the input audio.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input audio as a NumPy array or PyTorch tensor.
            sr (int): Sample rate of the audio.

        Returns:
            torch.Tensor: VAD output tensor.

        Raises:
            ValueError: If sample rate is unsupported.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            logger.debug("Converted input audio from NumPy array to torch.Tensor.")

        if not isinstance(x, torch.Tensor):
            logger.error("Input audio must be a NumPy array or torch.Tensor.")
            raise TypeError("Input audio must be a NumPy array or torch.Tensor.")

        x, sr = self._validate_input(x, sr)
        batch_size = x.shape[0]

        if (
            self._last_batch_size != batch_size
            or self._last_sr != sr
        ):
            self.reset_states(batch_size)
            logger.debug(
                f"Batch size or sample rate changed. Reset states to batch size: {batch_size}, sample rate: {sr}."
            )

        ort_inputs = {
            "input": x.numpy(),
            "h": self._h,
            "c": self._c,
            "sr": np.array(sr, dtype=np.int64),
        }

        try:
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
            logger.debug("VAD inference completed successfully.")
        except onnxruntime.OrtException as e:
            logger.error(f"ONNX Runtime inference failed: {e}")
            raise RuntimeError(f"ONNX Runtime inference failed: {e}")

        out = torch.tensor(out)
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

    def audio_forward(
        self, x: Union[np.ndarray, torch.Tensor], sr: int, num_samples: int = 512
    ) -> torch.Tensor:
        """
        Processes audio in chunks and performs VAD on each chunk.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input audio as a NumPy array or PyTorch tensor.
            sr (int): Sample rate of the audio.
            num_samples (int, optional): Number of samples per chunk. Defaults to 512.

        Returns:
            torch.Tensor: Concatenated VAD output tensor.
        """
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples != 0:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = F.pad(x, (0, pad_num), "constant", 0.0)
            logger.debug(f"Padded input audio with {pad_num} samples.")

        self.reset_states(x.shape[0])
        logger.debug(f"Processing audio in chunks of {num_samples} samples.")

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)
            logger.debug(f"Processed chunk {i // num_samples + 1}.")

        if outs:
            stacked = torch.cat(outs, dim=1)
            logger.debug("Concatenated all VAD output chunks.")
            return stacked.cpu()
        else:
            logger.warning("No audio chunks were processed.")
            return torch.tensor([])


def load_vad(path: Optional[str] = None, download: bool = True) -> SileroVAD:
    """
    Loads the Silero VAD model, downloading it if necessary.

    Args:
        path (Optional[str], optional): Path to the VAD model file. Defaults to None.
        download (bool, optional): Whether to download the model if not found. Defaults to True.

    Returns:
        SileroVAD: An instance of the SileroVAD class.

    Raises:
        RuntimeError: If the model file is not found and download is disabled, or if the MD5 checksum does not match.
    """
    if path is None:
        make_cache_dir()
        cache_dir = get_cache_dir()
        path = cache_dir / SILERO_VAD_ONNX_FILENAME
        logger.debug(f"No path provided. Using cache directory: {cache_dir}")

    path = Path(path)

    if not path.is_file():
        if not download:
            logger.error(
                f"VAD model not found at '{path}'. Please set download=True to download it."
            )
            raise RuntimeError(
                f"VAD model not found at '{path}'. Please set download=True to download it."
            )
        logger.info(f"VAD model not found at '{path}'. Downloading from '{SILERO_VAD_ONNX_URL}'.")
        download_file(str(SILERO_VAD_ONNX_URL), str(path), makedirs=True)

    if not check_file_md5(str(path), SILERO_VAD_ONNX_MD5_CHECKSUM):
        logger.error(
            f"The MD5 checksum for '{path}' does not match the expected value."
        )
        raise RuntimeError(
            f"The MD5 checksum for '{path}' does not match the expected value."
        )

    logger.info(f"Loading Silero VAD model from '{path}'.")
    return SileroVAD(path)
