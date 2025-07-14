import logging.config
import sys
import os
import json
def setup_logging(log_path=None, level=logging.INFO, logs_config_path=None):
    """
    Set up logging using a config file if provided, else use defaults.
    """
    config = None
    if logs_config_path and os.path.exists(logs_config_path):
        with open(logs_config_path, 'r') as f:
            config = json.load(f)
    if config:
        # Build handlers
        handlers = {}
        if config.get("log_to_file", True):
            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": config.get("level", "INFO"),
                "formatter": "default",
                "filename": config.get("log_file_path", "logs/logs.txt"),
                "maxBytes": config.get("max_bytes", 1048576),
                "backupCount": config.get("backup_count", 3)
            }
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": config.get("level", "INFO"),
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": config.get("format", "%(asctime)s %(levelname)s %(message)s")
                }
            },
            "handlers": handlers,
            "root": {
                "level": config.get("level", "INFO"),
                "handlers": list(handlers.keys())
            }
        }
        logging.config.dictConfig(logging_config)
    else:
        # Fallback to default
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_path:
            handlers.append(logging.FileHandler(log_path))
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=handlers
        )