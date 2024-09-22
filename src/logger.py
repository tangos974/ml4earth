# logger.py
import logging
import os
import sys

from colorlog import ColoredFormatter


def supports_color():
    """
    Check if the environment supports color output.
    """
    supported_platform = sys.platform != "win32" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty


def setup_logger(
    output_folder: str,
    run_id: str,
    logs_folder: str = "logs",
):
    """
    Set up the logger to write to both the console and a file.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_file_name (str): Name of the log file.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Full path to the logs folder
    full_log_dir = os.path.join(output_folder, logs_folder)

    # Ensure the output and logs directories exist
    os.makedirs(full_log_dir, exist_ok=True)

    log_file_path = os.path.join(full_log_dir, f"{run_id}.log")

    # Set up logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG)

    # Formatter to include timestamps and logging level
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter_default = logging.Formatter(log_format)

    # File handler to write to a log file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter_default)
    file_handler.setLevel(logging.INFO)

    # Use color only in the terminal
    formatter_color = ColoredFormatter(
        "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # Console handler to print to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter_color)
    console_handler.setLevel(logging.DEBUG)

    # Adding both handlers to the logger
    if not logger.handlers:  # Avoid adding multiple handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
