import configparser
import app_logger
from detect_detctron2 import DetectDetectron

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read('config/basic.ini')



def main() -> None:
    
    dd = DetectDetectron()
    dd.run_predictions('/vulcan/scratch/vshenoy/aicity2020/dataset_A_frames/cam_10')



if __name__ == '__main__':

    main()