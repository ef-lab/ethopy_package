# ethopy/run.py
import logging
import sys
import time
import traceback

from ethopy.core.logger import Logger
from ethopy.utils.start import PyWelcome

log = logging.getLogger(__name__)  # Get logger for this module


def run(protocol=False):
    logger = Logger(protocol=protocol)

    # # # # Waiting for instructions loop # # # # #
    while logger.setup_status != 'exit':
        if logger.setup_status != 'running':
            log.info("################ EthoPy Welcome ################")
            PyWelcome(logger)
        if logger.setup_status == 'running':   # run experiment unless stopped
            try:
                if logger.get_protocol():
                    namespace = {"logger": logger}
                    exec(open(logger.protocol_path, encoding="utf-8").read(), namespace)
            except Exception as e:
                log.error("ERROR: %s", traceback.format_exc())
                logger.update_setup_info({'state': 'ERROR!', 'notes': str(e), 'status': 'exit'})
            if logger.manual_run:
                logger.update_setup_info({'status': 'exit'})
                break
            elif logger.setup_status not in ['exit', 'running']:  # restart if session ended
                logger.update_setup_info({'status': 'ready'})  # restart
        time.sleep(.1)

    logger.cleanup()
    sys.exit(0)
