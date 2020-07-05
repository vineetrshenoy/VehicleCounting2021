import sys
import configparser
import app_logger
#from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_detector import VisualizeDetector
from visualize_tracker import VisualizeTracker
from bezier_matching import BezierMatching
from visualize_counting import VisualizeCounting
from effdet import EfficientDet
import pickle
logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])



def main() -> None:
    
    dt = EfficientDet()
    detection_dict_list = dt.generate_predictions()
    Tracker().run_tracker()
    VisualizeTracker().run_visualizations()
    BezierMatching().workflow()
    VisualizeCounting().workflow()
if __name__ == '__main__':

    main()