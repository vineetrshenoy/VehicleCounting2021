import sys
import configparser
import app_logger
from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_detector import VisualizeDetector
from visualize_tracker import VisualizeTracker
from bezier_matching import BezierMatching
from visualize_counting import VisualizeCounting

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])



def main() -> None:
    
    #dt = DetectDetectron()
    #detection_dict = dt.run_predictions()

    '''
    if int(config['DETECTION']['visualize']) == 1:
        VisualizeDetector('cam_1', 10, (1280, 960)).run_visualizations()
    '''


    #Tracker().run_tracker(detection_dict)
    #VisualizeTracker().run_visualizations()
    BezierMatching().workflow()
    VisualizeCounting().workflow()
if __name__ == '__main__':

    main()