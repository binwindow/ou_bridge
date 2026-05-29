import json
import logging
import os
import sys
import time


def setup_logger(log_file: str) -> logging.Logger:
    """Set up a logger that mirrors terminal output to a file."""
    logger = logging.getLogger("goub")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class JSONLMetrics:
    """Append-only JSONL metrics writer."""

    def __init__(self, path: str):
        self.path = path
        self._file = None

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "a")

    def write(self, record: dict):
        if self._file is None:
            self.open()
        record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


class Logger:
    """Training logger: writes JSONL metrics + mirrors to terminal via logging."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.py_logger = setup_logger(os.path.join(log_dir, "train.log"))
        self.train_metrics = JSONLMetrics(os.path.join(log_dir, "train_metrics.jsonl"))
        self.val_metrics = JSONLMetrics(os.path.join(log_dir, "val_metrics.jsonl"))

    def info(self, msg: str):
        self.py_logger.info(msg)

    def log_train(self, record: dict):
        record["type"] = "train"
        self.train_metrics.write(record)

    def log_val(self, record: dict):
        record["type"] = "val"
        self.val_metrics.write(record)

    def save_config(self, config_dict: dict):
        path = os.path.join(self.log_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def save_parameter_info(self, model_info: dict):
        path = os.path.join(self.log_dir, "parameter.json")
        with open(path, "w") as f:
            json.dump(model_info, f, indent=2)

    def close(self):
        self.train_metrics.close()
        self.val_metrics.close()
