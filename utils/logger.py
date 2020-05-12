import logging
import sys
import os


class Logger:
    def __init__(self, logger_name, output_name):
        self.logger_name = logger_name
        self.output_name = output_name

        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        stderr_handler = logging.StreamHandler(sys.stderr)
        file_handler = logging.FileHandler(self.output_name)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stderr_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        logger.addHandler(file_handler)
        return logger

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.exception(msg, *args, exc_info, **kwargs)

