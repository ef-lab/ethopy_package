"""Main execution module for EthoPy experiments.

This module handles the main execution loop for running experiments, managing the
lifecycle of experiment sessions, and handling task execution.
"""

import logging
import sys
import time
import traceback
from typing import Optional

from ethopy.core.logger import Logger
from ethopy.utils.start import PyWelcome
from ethopy.utils.task import Task

log = logging.getLogger(__name__)  # Get logger for this module


def run(task: Optional[Task] = None) -> None:
    """Run the main execution loop for the experiment."""
    logger = Logger(task=task)

    # # # # Waiting for instructions loop # # # # #
    while logger.setup_status != "exit":
        if logger.setup_status != "running":
            log.info("################ EthoPy Welcome ################")
            PyWelcome(logger)
        if logger.setup_status == "running":  # run experiment unless stopped
            try:
                if logger.get_task():
                    namespace = {"logger": logger}
                    exec(open(logger.task_path, encoding="utf-8").read(), namespace)
            except Exception as e:
                log.error("ERROR: %s", traceback.format_exc())
                logger.update_setup_info(
                    {"state": "ERROR!", "notes": str(e), "status": "exit"}
                )
            if logger.manual_run:
                logger.update_setup_info({"status": "exit"})
                break
            elif logger.setup_status not in [
                "exit",
                "running",
            ]:  # restart if session ended
                logger.update_setup_info({"status": "ready"})  # restart
        time.sleep(0.1)

    logger.cleanup()
    sys.exit(0)
