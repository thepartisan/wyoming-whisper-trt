import hashlib
from pathlib import Path
from typing import Optional

import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


# Configure logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Constants
SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx"
)
SILERO_VAD_ONNX_FILENAME = "silero_vad.onnx"
SILERO_VAD_ONNX_MD5_CHECKSUM = "03da8de2fec4108a089b39f1b4abefef"

SUPPORTED_LANGUAGES = ['ru', 'en', 'de', 'es']

__all__ = [
    "SileroVAD",
    "load_vad"
]


def download_file(
    url: str,
    path: str,
    makedirs: bool = False,
    timeout: Optional[int] = 30,
    retries: int = 3,
    backoff_factor: float = 0.3
) -> None:
    """
    Download a file from a given URL to a specified path with retry and progress indication.

    Args:
        url (str): The URL to download the file from.
        path (str): The destination file path where the downloaded file will be saved.
        makedirs (bool, optional): If True, creates the parent directories if they do not exist. Defaults to False.
        timeout (Optional[int], optional): Timeout for the HTTP request in seconds. Defaults to 30.
        retries (int, optional): Number of retries for failed downloads. Defaults to 3.
        backoff_factor (float, optional): Backoff factor for retries. Defaults to 0.3.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        Exception: For other exceptions during the download process.
    """
    destination = Path(path)

    if makedirs:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories up to: {destination.parent}")
        except Exception as e:
            logger.error(f"Failed to create directories for {destination}: {e}")
            raise

    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        logger.info(f"Starting download from {url} to {destination}")
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KB
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            with destination.open('wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            error_msg = f"Downloaded size mismatch: {progress_bar.n} != {total_size_in_bytes}"
            logger.error(error_msg)
            raise IOError(error_msg)

        logger.info(f"Successfully downloaded {url} to {destination}")
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred while downloading {url}: {http_err}")
        raise
    except Exception as err:
        logger.error(f"An error occurred while downloading {url}: {err}")
        raise


def check_file_md5(path: str, target_md5: str, chunk_size: int = 8192) -> bool:
    """
    Check if the MD5 checksum of a file matches the target checksum.

    Args:
        path (str): Path to the file to be checked.
        target_md5 (str): The target MD5 checksum to compare against.
        chunk_size (int, optional): Size of each chunk to read from the file. Defaults to 8192.

    Returns:
        bool: True if the file's MD5 matches the target, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other exceptions during the checksum calculation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"No such file: '{file_path}'")

    hash_md5 = hashlib.md5()
    try:
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash_md5.update(chunk)
        computed_md5 = hash_md5.hexdigest()
        if computed_md5.lower() == target_md5.lower():
            logger.debug(f"MD5 checksum matched for {file_path}: {computed_md5}")
            return True
        else:
            logger.warning(f"MD5 checksum mismatch for {file_path}: {computed_md5} != {target_md5}")
            return False
    except Exception as e:
        logger.error(f"Error computing MD5 for {file_path}: {e}")
        raise


class SileroVAD:
    """
    Silero Voice Activity Detection (VAD) using an ONNX model.
    """

    SAMPLE_RATES = [8000, 16000]
    HIDDEN_SIZE = 64
    DEFAULT_BATCH_SIZE = 1

    def __init__(self, path: str, force_cpu: bool = False) -> None:
        """
        Initialize the SileroVAD with the specified ONNX model.

        Args:
            path (str): Path to the Silero VAD ONNX model file.
            force_cpu (bool, optional): If True, forces the use of CPU for inference. Defaults to False.
        """
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        providers = ['CPUExecutionProvider'] if force_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers() else None

        try:
            self.session = onnxruntime.InferenceSession(path, sess_options=session_options, providers=providers)
            logger.debug(f"Initialized ONNX Runtime session with providers: {providers or 'default'}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise RuntimeError(f"ONNX Runtime session initialization failed: {e}") from e

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
            batch_size (int, optional): The batch size for the VAD model. Defaults to DEFAULT_BATCH_SIZE.
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
        if (self._last_sr is not None and self._last_sr != sr) or \
           (self._last_batch_size is not None and self._last_batch_size != batch_size):
            logger.debug("Sample rate or batch size changed. Resetting states.")
            self.reset_states(batch_size=batch_size)

        self._last_sr = sr
        self._last_batch_size = batch_size

        ort_inputs = {
            'input': x.numpy(),
            'h': self._h,
            'c': self._c,
            'sr': np.array(sr, dtype=np.int64)
        }

        try:
            ort_outputs = self.session.run(None, ort_inputs)
            logger.debug("ONNX Runtime inference successful.")
        except Exception as e:
            logger.error(f"ONNX Runtime inference failed: {e}")
            raise RuntimeError(f"ONNX Runtime inference failed: {e}") from e

        out, self._h, self._c = ort_outputs
        logger.debug(f"Output shape: {out.shape}, Updated hidden states shape: {self._h.shape}, {self._c.shape}")

        return torch.tensor(out)

    def audio_forward(self, x: torch.Tensor, sr: int, num_samples: int = 512) -> torch.Tensor:
        """
        Perform VAD on audio by processing it in chunks.

        Args:
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the input audio.
            num_samples (int, optional): Number of samples per chunk. Defaults to 512.

        Returns:
            torch.Tensor: Concatenated VAD outputs.
        """
        logger.debug(f"Starting audio_forward with num_samples={num_samples}")
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples != 0:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), 'constant', 0.0)
            logger.debug(f"Padded input audio with {pad_num} zeros to make it divisible by {num_samples}.")

        self.reset_states(batch_size=x.size(0))
        logger.debug(f"Processing audio in chunks of {num_samples} samples.")

        for i in range(0, x.size(1), num_samples):
            wavs_batch = x[:, i:i + num_samples]
            logger.debug(f"Processing chunk {i // num_samples + 1}: samples {i} to {i + num_samples}")
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1).cpu()
        logger.debug(f"Concatenated output shape: {stacked.shape}")
        return stacked


def load_vad(
    path: Optional[str] = None,
    download: bool = True,
    force_cpu: bool = False
) -> SileroVAD:
    """
    Load the Silero VAD model.

    Args:
        path (Optional[str], optional): Path to the Silero VAD ONNX model. If None, it will be downloaded to the cache directory. Defaults to None.
        download (bool, optional): Whether to download the model if it's not found at the specified path. Defaults to True.
        force_cpu (bool, optional): If True, forces the use of CPU for inference. Defaults to False.

    Returns:
        SileroVAD: An instance of the SileroVAD class.

    Raises:
        RuntimeError: If the model file is not found and downloading is disabled, or if the MD5 checksum does not match.
    """
    if path is None:
        make_cache_dir()
        path = str(Path(get_cache_dir()) / SILERO_VAD_ONNX_FILENAME)
        logger.debug(f"No model path provided. Using cache path: {path}")
    else:
        path = str(Path(path))
        logger.debug(f"Using provided model path: {path}")

    if not Path(path).exists():
        if not download:
            error_msg = f"VAD model not found at {path}. Please set download=True to download it."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"Model not found at {path}. Downloading from {SILERO_VAD_ONNX_URL}")
        download_file(SILERO_VAD_ONNX_URL, path, makedirs=True)
        logger.info(f"Model downloaded to {path}")

    if not check_file_md5(path, SILERO_VAD_ONNX_MD5_CHECKSUM):
        error_msg = f"The MD5 checksum for {path} does not match the expected value."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.debug(f"MD5 checksum for {path} verified successfully.")

    return SileroVAD(path, force_cpu=force_cpu)
