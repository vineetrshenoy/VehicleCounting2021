import sys
import configparser
import app_logger
from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_detector import VisualizeDetector
from visualize_tracker import VisualizeTracker

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])



def main() -> None:
    
    dt = DetectDetectron('cam_1', 10, (1920, 1080))
    detection_dict = dt.run_predictions()

    if int(config['DETECTION']['visualize']) == 1:
        VisualizeDetector('cam_1', 10, (1280, 960)).run_visualizations()
    
    tr = Tracker('cam_1')
    tr.run_tracker(detection_dict)
    VisualizeTracker('cam_1', 10, (1280, 960)).run_visualizations()


if __name__ == '__main__':

    main()