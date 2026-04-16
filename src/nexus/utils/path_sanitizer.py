"""
utils/path_sanitizer.py

Windows-compatible path sanitization utility.
Resolves [Errno 22] Invalid argument errors caused by invalid characters or path length.
"""

import os
import re
from pathlib import Path
from typing import Union

# Characters illegal in Windows file names
WINDOWS_ILLEGAL_CHARS = r'[<>:"/\\|?*]'
MAX_WINDOWS_PATH_LENGTH = 240  # Conservative limit (Windows max is 260)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for Windows compatibility.

    Replaces illegal characters (<>:"/\\|?*) with underscores.
    Does NOT handle full paths - use sanitize_path() for that.

    Args:
        filename: The filename to sanitize (not a full path)

    Returns:
        Sanitized filename safe for Windows
    """
    # Replace illegal characters with underscore
    sanitized = re.sub(WINDOWS_ILLEGAL_CHARS, '_', filename)

    # Remove leading/trailing spaces and dots (Windows doesn't like them)
    sanitized = sanitized.strip(' .')

    # Ensure filename isn't empty after sanitization
    if not sanitized:
        sanitized = "_unnamed_"

    return sanitized


def sanitize_path(path: Union[str, Path], enforce_length: bool = True) -> str:
    """
    Sanitize a full file path for Windows compatibility.

    - Replaces illegal characters in filename component
    - Optionally truncates to MAX_WINDOWS_PATH_LENGTH if too long
    - Preserves directory structure and file extension

    Args:
        path: Full file path to sanitize
        enforce_length: If True, truncate paths exceeding Windows limits

    Returns:
        Sanitized path string safe for Windows
    """
    path_str = str(path)

    # Split into directory and filename
    directory = os.path.dirname(path_str)
    filename = os.path.basename(path_str)

    # Sanitize just the filename
    name, ext = os.path.splitext(filename)
    sanitized_name = sanitize_filename(name)
    sanitized_filename = sanitized_name + ext

    # Reconstruct full path
    sanitized_path = os.path.join(directory, sanitized_filename) if directory else sanitized_filename

    # Enforce length limit if requested
    if enforce_length and len(sanitized_path) > MAX_WINDOWS_PATH_LENGTH:
        # Truncate the filename portion, preserving extension
        available_length = MAX_WINDOWS_PATH_LENGTH - len(directory) - len(ext) - 1  # -1 for separator
        if available_length < 10:
            # Directory path is too long, just truncate everything
            available_length = MAX_WINDOWS_PATH_LENGTH - len(ext)
            sanitized_path = sanitized_path[:available_length] + ext
        else:
            truncated_name = sanitized_name[:available_length]
            sanitized_path = os.path.join(directory, truncated_name + ext)

    return sanitized_path


def sanitize_timestamp_for_filename(timestamp: str) -> str:
    """
    Sanitize an ISO timestamp for use in filenames.

    ISO timestamps contain colons (e.g., 2026-02-07T07:15:07) which are
    illegal in Windows filenames. This replaces them with hyphens.

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Filename-safe timestamp string
    """
    # Replace colons with hyphens (common convention)
    return timestamp.replace(':', '-')


def ensure_safe_open(filepath: Union[str, Path], mode: str = 'r', **kwargs):
    """
    Context manager that sanitizes path before opening file.

    Usage:
        with ensure_safe_open(path, 'w') as f:
            f.write(data)

    Args:
        filepath: Path to file
        mode: File open mode
        **kwargs: Additional arguments passed to open()

    Returns:
        File handle from open()
    """
    safe_path = sanitize_path(filepath)
    return open(safe_path, mode, **kwargs)
