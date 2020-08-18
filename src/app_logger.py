import logging
import logging.handlers
import os
from pathlib import Path
import sys
import configparser

config = configparser.ConfigParser()
config.read('config/basic.ini')


# create file handler which logs even debug messages
log_name = os.path.join('logs', config['DEFAULT']['log_filename'])
Path(log_name).touch(exist_ok=True)
fh = logging.handlers.TimedRotatingFileHandler(log_name, when='H', backupCount=10)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s.%(name)s] - %(message)s')
fh.setFormatter(formatter)

def get_logger(module_name: str):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # add the handlers to the logger
    
    logger.addHandler(fh)
    
    return logger