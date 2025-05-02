# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import logging.config
import logging.handlers
import threading
import yaml
from pathlib import Path

from accelerate import PartialState

from .schema.logging import LogConfig, Level


enable_log_buffering = False


class ThreadLocalLogBuffer:
    """Thread-local log buffer for buffering log messages until the accelerator is initialized.

    This class provides a thread-local buffer for storing log messages until the accelerator
    is initialized. When the accelerator is initialized, the buffer is flushed, and the log
    messages are written to the logger.

    Attributes:
        _instance (ThreadLocalLogBuffer): Singleton instance of the ThreadLocalLogBuffer class.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_buffer = ThreadLocalLogBuffer.get_instance(logger)
        >>> log_buffer.log(logging.INFO, "This is an info message")
        >>> # Flush the buffer
        >>> log_buffer.flush_buffer()
    """

    _instance = None

    def __init__(self, logger):
        """Initialize the ThreadLocalLogBuffer instance.

        Args:
            logger (logging.Logger): The logger instance to use for logging.
        """
        self.logger = logger
        self._thread_local = threading.local()
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Initialize the thread-local buffer if it doesn't exist."""
        if not hasattr(self._thread_local, "buffer"):
            self._thread_local.buffer = []

    def log(self, level, msg, *args, **kwargs):
        """Log a message to the buffer.

        Args:
            level (int): The logging level of the message.
            msg (str): The log message.
            *args: Positional arguments to be passed to the logger.
            **kwargs: Keyword arguments to be passed to the logger.
        """
        self._initialize_buffer()
        self._thread_local.buffer.append((level, msg, args, kwargs))

    def flush_buffer(self):
        """Flush the buffer by logging all messages to the logger."""
        if hasattr(self._thread_local, "buffer"):
            for level, msg, args, kwargs in self._thread_local.buffer:
                self.logger.log(level, msg, *args, **kwargs)
            self._thread_local.buffer = []

    @classmethod
    def get_instance(cls, logger):
        """Get the singleton instance of the ThreadLocalLogBuffer class.

        Args:
            logger (logging.Logger): The logger instance to use for logging.

        Returns:
            ThreadLocalLogBuffer: The singleton instance of the ThreadLocalLogBuffer class.
        """
        if cls._instance is None:
            cls._instance = cls(logger)
        return cls._instance


class MultiProcessAdapter(logging.LoggerAdapter):
    """HuggingFace Accelerate-compatible logger for handling multi-process logging.

    This class is a LoggerAdapter that handles multi-process logging and is compatible
    with HuggingFace Accelerate. It buffers log messages until the accelerator is initialized
    and flushes the buffer when the accelerator is initialized or when the instance is destroyed.

    Attributes:
        log_buffer (ThreadLocalLogBuffer): The thread-local log buffer instance.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> multi_process_logger = MultiProcessAdapter(logger)
        >>> multi_process_logger.log(logging.INFO, "This is an info message")
    """

    def __init__(self, logger):
        """Initialize the MultiProcessAdapter instance.

        Args:
            logger (logging.Logger): The logger instance to adapt.
        """
        super().__init__(logger, {})
        self.log_buffer = ThreadLocalLogBuffer.get_instance(logger)

    def __del__(self):
        # Flush the buffer when the instance is being destroyed, necessary for commands without the accelerator state
        if not self._accelerator_is_initialized():
            self.log_buffer.flush_buffer()

    @staticmethod
    def _should_log(main_process_only):
        """Check if the current process should log.

        Args:
            main_process_only (bool): Whether to log only in the main process.

        Returns:
            bool: True if the current process should log, False otherwise.
        """
        return not main_process_only or (main_process_only and PartialState().is_main_process)

    @staticmethod
    def _accelerator_is_initialized():
        """Check if the accelerator is initialized.

        Returns:
            bool: True if the accelerator is initialized, False otherwise.
        """
        return not PartialState._shared_state == {}

    def log(self, level, msg, *args, **kwargs):
        """Log a message with the given level, message, and arguments.

           Buffers logs when invoked with accelerator to avoid duplicate logs using the main process.

        Args:
            level (int): The logging level of the message.
            msg (str): The log message.
            *args: Positional arguments to be passed to the logger.
            **kwargs: Keyword arguments to be passed to the logger.
        """
        global enable_log_buffering
        if not enable_log_buffering:
            self.logger.log(level, msg, *args, **kwargs)
            return

        if not self._accelerator_is_initialized():
            # Accelerate usually has a RuntimeError when logging for uninitialized accelerate. Instead, we want to log
            # normally to defer creating the Accelerator() instance on CLI startup. This means we buffer logs if
            # num_processes>1 for the first logs in the CLI commands and until Accelerate() instance the created.
            # Once created, we flush the buffers to the loggers for the main process.
            self.log_buffer.log(level, msg, *args, **kwargs)
            return

        # Accelerate internally passes `main_process_only` to log statements
        main_process_only = kwargs.pop("main_process_only", True)

        # Accelerate is initialized, flush the log buffer
        if self._should_log(main_process_only):
            self.log_buffer.flush_buffer()

        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)


def getLogger(name: str):
    """Get a multi-process logger instance. Set options on the logger via :func:`configure_logging`.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.LoggerAdapter: A multi-process logger instance.
    """
    return MultiProcessAdapter(logging.getLogger(name))


class FileHandler(logging.handlers.RotatingFileHandler):
    """Custom file handler that lazily creates directories and files.

    This file handler creates missing directories and files lazily when logging. It also
    supports log rotation and can handle multi-process logging with HuggingFace Accelerate.

    Attributes:
        filename (str): The path of the log file to write to.
        mkdirs (bool): Whether to create missing directories in the path.
        exist_ok (bool): Whether to suppress errors if log directories exist.
        rotate (bool): Whether to enable log rotation.
        rotateOnExists (bool): Whether to rotate the log file if it already exists.
        do_rotate_on_exists (bool): Whether to rotate the log file on the first emit.
    """

    def __init__(
        self,
        filename: str,
        mkdirs: bool = True,
        exist_ok: bool = True,
        rotate: bool = False,
        rotateOnExists: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the handler.

        Args:
            filename: Path of logfile to write to.
            mkdirs: Whether to create missing directories in path.
            exist_ok: Whether to suppress errors if log directories exist.
            rotate: Whether to enable log rotation.
        """
        self.filename = filename
        self.mkdirs = mkdirs
        self.exist_ok = exist_ok
        self.rotateOnExists = rotateOnExists
        self.do_rotate_on_exists = False

        super().__init__(filename, delay=True, *args, **kwargs)

        if self.rotateOnExists and Path(self.filename).exists():
            # lazily rotate in emit() for multi-process logging
            self.do_rotate_on_exists = True

        if not rotate and not self.do_rotate_on_exists:
            self._disable_rotate()

    def _disable_rotate(self):
        self.backupCount = 0
        self.maxBytes = 0

    def emit(self, record):
        """Emit a log record, creating directories/files lazily."""
        Path(self.filename).parent.mkdir(parents=self.mkdirs, exist_ok=self.exist_ok)

        # Handle multi-process accelerate.logging() by rolloving here rather than during __init__
        # to give accelerate a chance to check _should_log(). This avoids erroneously
        # rolling over when more than one process configure logging at the same time/on start.
        if self.do_rotate_on_exists:
            self.doRollover()
            self.do_rotate_on_exists = False
            self._disable_rotate()

        super().emit(record)

    def rotate(self, source, dest):
        """Override rotate to create directories."""
        Path(dest).parent.mkdir(parents=self.mkdirs, exist_ok=self.exist_ok)
        super().rotate(source, dest)


def configure_logging(log_config: LogConfig, use_log_buffering: bool = False) -> None:
    """
    Configure logging from a LogConfig object.
    If a .yaml file is not specified, uses basic logging.
    """

    # hack: allow disabling buffering when accelerate is not used
    global enable_log_buffering
    enable_log_buffering = use_log_buffering

    if log_config.use_config_file is False:
        logging.basicConfig(
            format="[%(levelname)s] - %(name)s - %(message)s",
            level=log_config.level if log_config.level else logging.INFO,
        )
    else:
        try:
            with log_config.config_file.open() as f:
                config = yaml.safe_load(f)

                # override root and console with --log-level if present, handle root.level key existence too
                default_level = logging.WARNING
                level = Level.name_mapping.get(config.get("root", {}).get("level", None), default_level)
                level = log_config.level if log_config.level else level

                if "root" in config:
                    config["root"]["level"] = level

                if "handlers" in config and "console" in config["handlers"]:
                    config["handlers"]["console"]["level"] = level

                # default to "logs"
                log_config.prefix_dir = Path(log_config.prefix_dir or "logs")

                # prepend prefix directory to all file handlers
                if "handlers" in config:
                    for handler in config["handlers"].values():
                        if handler["class"] == "mlsimkit.common.logging.FileHandler":
                            handler["filename"] = log_config.prefix_dir / handler["filename"]

                logging.config.dictConfig(config)
        except Exception as e:
            raise Exception(f"Error loading config file '{log_config.config_file}'", e)
