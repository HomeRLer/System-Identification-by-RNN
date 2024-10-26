import os
import shutil
import sys

from data.utils import SimpleLogger


def config_logging(output_dir: str = "output") -> SimpleLogger:
    log_dir = os.path.join(output_dir, "log.txt")
    if os.path.isfile(log_dir):
        with open(log_dir, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            completed = "Finished" in content
            if completed:
                sys.exit()
            else:
                shutil.rmtree(output_dir, ignore_errors=True)

    logging = SimpleLogger(log_dir)
    return logging
