import logging
import logging.handlers
import os

import sys


# create file handler which logs even debug messages
fh = logging.handlers.TimedRotatingFileHandler('app.log', when='H', backupCount=10)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s.%(name)s] - %(message)s')
fh.setFormatter(formatter)

def get_logger(module_name: str):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # add the handlers to the logger
    
    logger.addHandler(fh)
    
    return logger