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
from typing import Optional
import os
import threading

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Thread lock for thread-safe operations
_cache_lock = threading.Lock()

# Default cache directory, can be overridden by environment variable
_DEFAULT_CACHE_DIR = Path(os.getenv("WHISPER_TRT_CACHE_DIR", Path.home() / ".cache" / "whisper_trt")).resolve()

# Initialize cache directory
_CACHE_DIR = _DEFAULT_CACHE_DIR


def get_cache_dir() -> Path:
    """
    Retrieves the current cache directory path.

    Returns:
        Path: The path to the cache directory.
    """
    return _CACHE_DIR


def make_cache_dir() -> None:
    """
    Creates the cache directory if it does not already exist.

    Raises:
        OSError: If the directory cannot be created due to permission issues or invalid paths.
    """
    with _cache_lock:
        if not _CACHE_DIR.exists():
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created cache directory at: {_CACHE_DIR}")
            except OSError as e:
                logger.error(f"Failed to create cache directory at {_CACHE_DIR}: {e}")
                raise


def set_cache_dir(path: str) -> None:
    """
    Sets a new cache directory path.

    Args:
        path (str): The new path to set as the cache directory.

    Raises:
        ValueError: If the provided path is not a valid directory path.
        OSError: If the directory cannot be created due to permission issues or invalid paths.
    """
    new_path = Path(path).expanduser().resolve()
    if not new_path.is_absolute():
        logger.error(f"Provided path '{path}' is not absolute.")
        raise ValueError("Cache directory path must be absolute.")

    with _cache_lock:
        global _CACHE_DIR
        _CACHE_DIR = new_path
        logger.info(f"Cache directory set to: {_CACHE_DIR}")
        make_cache_dir()


def cache_exists() -> bool:
    """
    Checks if the cache directory exists.

    Returns:
        bool: True if the cache directory exists, False otherwise.
    """
    return _CACHE_DIR.exists()


def get_cache_path(filename: str) -> Path:
    """
    Constructs the full path for a given cache filename.

    Args:
        filename (str): The name of the file within the cache directory.

    Returns:
        Path: The full path to the specified cache file.
    """
    return _CACHE_DIR / filename


def list_cache_files() -> list:
    """
    Lists all files in the cache directory.

    Returns:
        list: A list of filenames present in the cache directory.
    """
    if not _CACHE_DIR.exists():
        logger.warning(f"Cache directory {_CACHE_DIR} does not exist.")
        return []
    return [f.name for f in _CACHE_DIR.iterdir() if f.is_file()]
