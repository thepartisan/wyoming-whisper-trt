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

import hashlib
import logging
from pathlib import Path
from typing import Optional
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
import threading

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Thread lock for thread-safe operations
_download_lock = threading.Lock()


def download_file(url: str, path: str, makedirs: bool = False, chunk_size: int = 1024) -> None:
    """
    Downloads a file from the specified URL to the given path.

    Args:
        url (str): The URL of the file to download.
        path (str): The destination file path where the downloaded file will be saved.
        makedirs (bool, optional): Whether to create parent directories if they do not exist. Defaults to False.
        chunk_size (int, optional): The size of each chunk to read during download. Defaults to 1024 bytes.

    Raises:
        ValueError: If the URL is empty or invalid.
        RequestException: If the HTTP request fails.
        OSError: If the file cannot be written due to permission issues or invalid paths.
    """
    if not url:
        logger.error("The download URL is empty.")
        raise ValueError("The download URL cannot be empty.")

    destination = Path(path)

    if makedirs:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured that the directory {destination.parent} exists.")
        except OSError as e:
            logger.error(f"Failed to create directories for path '{destination}': {e}")
            raise

    try:
        with _download_lock:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                logger.info(f"Starting download from {url} to {destination}. Total size: {total_size} bytes.")

                with open(destination, 'wb') as file, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=destination.name
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                            progress_bar.update(len(chunk))
        logger.info(f"Successfully downloaded '{url}' to '{destination}'.")
    except RequestException as e:
        logger.error(f"Failed to download '{url}': {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to write to '{destination}': {e}")
        raise


def check_file_md5(path: str, target: str) -> bool:
    """
    Checks whether the MD5 checksum of the file at the given path matches the target checksum.

    Args:
        path (str): The path to the file to check.
        target (str): The target MD5 checksum to compare against.

    Returns:
        bool: True if the file's MD5 checksum matches the target, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        OSError: If the file cannot be read due to permission issues or other I/O errors.
    """
    file_path = Path(path)

    if not file_path.is_file():
        logger.error(f"The file '{path}' does not exist.")
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    try:
        hash_md5 = hashlib.md5()
        total_size = file_path.stat().st_size
        with file_path.open('rb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=f"Computing MD5 for {file_path.name}"
        ) as progress_bar:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
                progress_bar.update(len(chunk))
        computed_md5 = hash_md5.hexdigest()
        if computed_md5 == target:
            logger.info(f"MD5 checksum for '{path}' matches the target.")
            return True
        else:
            logger.warning(f"MD5 checksum for '{path}' does not match the target. Computed: {computed_md5}, Target: {target}")
            return False
    except OSError as e:
        logger.error(f"Failed to read file '{path}' for MD5 computation: {e}")
        raise


def get_filename_from_url(url: str) -> str:
    """
    Extracts the filename from a given URL.

    Args:
        url (str): The URL from which to extract the filename.

    Returns:
        str: The extracted filename.

    Raises:
        ValueError: If the URL does not contain a valid filename.
    """
    if not url:
        logger.error("The URL is empty.")
        raise ValueError("The URL cannot be empty.")

    path = Path(requests.utils.urlparse(url).path)
    if not path.name:
        logger.error(f"No valid filename found in URL '{url}'.")
        raise ValueError(f"No valid filename found in URL '{url}'.")

    filename = path.name
    logger.debug(f"Extracted filename '{filename}' from URL '{url}'.")
    return filename


def compute_md5(path: str, chunk_size: int = 4096) -> str:
    """
    Computes the MD5 checksum of the file at the given path.

    Args:
        path (str): The path to the file.
        chunk_size (int, optional): The size of each chunk to read during computation. Defaults to 4096 bytes.

    Returns:
        str: The computed MD5 checksum in hexadecimal format.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        OSError: If the file cannot be read due to permission issues or other I/O errors.
    """
    file_path = Path(path)

    if not file_path.is_file():
        logger.error(f"The file '{path}' does not exist.")
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    try:
        hash_md5 = hashlib.md5()
        total_size = file_path.stat().st_size
        with file_path.open('rb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=f"Computing MD5 for {file_path.name}"
        ) as progress_bar:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
                progress_bar.update(len(chunk))
        computed_md5 = hash_md5.hexdigest()
        logger.debug(f"Computed MD5 for '{path}': {computed_md5}")
        return computed_md5
    except OSError as e:
        logger.error(f"Failed to read file '{path}' for MD5 computation: {e}")
        raise
