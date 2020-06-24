import sys
import configparser
import app_logger
from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_detector import VisualizeDetector
from visualize_tracker import VisualizeTracker
from bezier_matching import BezierMatching
from visualize_counting import VisualizeCounting
import pickle
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
    with open('/vulcan/scratch/vshenoy/vehicle_counting/src/vc_outputs/aicity/detection_output/cam_1/cam_1.pkl', 'rb') as f:
        detection_dict = pickle.load(f)

    Tracker().run_tracker(detection_dict)
    VisualizeTracker().run_visualizations()
    #BezierMatching().workflow()
    #VisualizeCounting().workflow()
if __name__ == '__main__':

    main()