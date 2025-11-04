import os
import sys
from pathlib import Path

from loguru import logger as _logger


MODULE_NAME = "reranker"
LOG_PATH = "logs"


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # 设置日志文件的最大大小为50MB
        self.log_file_size = "50 MB"
        # 日志文件保留的份数
        self.log_file_backup = 30
        self.formater = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | " \
                        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | " \
                        "\n<level>{message}</level>"

        _logger.remove()

        _logger.add(sink=sys.stderr, level="TRACE", diagnose=True, format=self.formater, colorize=True)

        _logger.add(os.path.join(self.log_dir, f"{MODULE_NAME}.log"),
                    rotation=self.log_file_size, retention=self.log_file_backup,
                    level="DEBUG", diagnose=True, format=self.formater, colorize=True,
                    enqueue=True, backtrace=True)

        _logger.add(os.path.join(self.log_dir, f"{MODULE_NAME}.log.trace"),
                    rotation=self.log_file_size, retention=self.log_file_backup,
                    level="TRACE", diagnose=True, format=self.formater,
                    enqueue=True, backtrace=True, filter=self.__trace_only)
        self._logger = _logger

    def __trace_only(self, record):
        return record["level"].name == "TRACE"

    def __debug_only(self, record):
        return record["level"].name == "DEBUG"

    def __success_only(self, record):
        return record["level"].name == "SUCCESS"

    def __warning_only(self, record):
        return record["level"].name == "WARNING"

    def __error_only(self, record):
        return record["level"].name == "ERROR"

    def get(self):
        return self._logger


Path(LOG_PATH).resolve().mkdir(parents=True, exist_ok=True)
logger = Logger(LOG_PATH).get()

if __name__ == '__main__':
    logger.info(f"hello {MODULE_NAME}!")
