import configparser
import app_logger
from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_tracker import VisualizeTracker

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read('config/basic.ini')



def main() -> None:
    
    dt = DetectDetectron('cam_8')
    detection_dict = dt.run_predictions()
    tr = Tracker('cam_8')
    tr.run_tracker(detection_dict)
    VisualizeTracker('cam_8').run_visualizations()


if __name__ == '__main__':

    main()