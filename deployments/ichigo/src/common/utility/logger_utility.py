import logging
from enum import Enum
from typing import ClassVar, Optional

from uvicorn.config import LOGGING_CONFIG


class LoggerUtility:
    """
    This class is used to create a logger object.
    """
    _logger: ClassVar[logging.Logger] = None

    class LogLevel(Enum):
        """
        This class is used to define the log level.
        """
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL

    @staticmethod
    def init_logger(name: str, log_level: LogLevel = LogLevel.INFO, log_file: Optional[str] = None) -> None:
        """
        This method is used to initialize the logger.
        """
        if LoggerUtility._logger is None:
            LoggerUtility._logger = logging.getLogger(name)
            LoggerUtility._logger.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                LoggerUtility._logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            LoggerUtility._logger.addHandler(console_handler)

            LOGGING_CONFIG["handlers"]["default"] = {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "default"
            }
            LOGGING_CONFIG["handlers"]["access"] = {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "access"
            }
            LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = log_level
            LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = log_level

    @staticmethod
    def get_logger() -> logging.Logger:
        """
        This method is used to create a logger object.
        """
        if LoggerUtility._logger is None:
            raise (Exception("Logger is not initialized."))
        else:
            return LoggerUtility._logger
