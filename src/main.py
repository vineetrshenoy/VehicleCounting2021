import sys
import configparser
import app_logger
#from detect_detctron2 import DetectDetectron
from tracker import Tracker
from visualize_detector import VisualizeDetector
from visualize_tracker import VisualizeTracker
from bezier_matching import BezierMatching
from visualize_counting import VisualizeCounting
from detection_tracker import DetectionTracker
#from deepsort_tracker import DeepsortTracker
from detection import DetectDali
import pickle
logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

def main() -> None:
    logger.info('JOB NAME {}'.format(basic_config['DEFAULT']['job_name']))
    
    #logger.info('Detection for {}'.format(config['DEFAULT']['cam_name']))
    #DetectDali().run_predictions()
    #
    #logger.info('Visualize Detection for {}'.format(config['DEFAULT']['cam_name']))
    #VisualizeDetector().run_visualizations()

    #DeepsortTracker().run_deepsort()
    #logger.info('Tracking for {}'.format(config['DEFAULT']['cam_name']))
    #Tracker().run_tracker()
    #
    #logger.info('Visualize Tracking for {}'.format(config['DEFAULT']['cam_name']))
    #VisualizeTracker().run_visualizations()
    #
    #logger.info('Counting for {}'.format(config['DEFAULT']['cam_name']))
    #BezierMatching().workflow()
    #
    #logger.info('Visualize Counting for {}'.format(config['DEFAULT']['cam_name']))
    #VisualizeCounting().workflow()

    DetectionTracker().workflow()
    #VisualizeDetector().run_visualizations()
    #VisualizeTracker().run_visualizations()
    #VisualizeCounting().workflow()



if __name__ == '__main__':

    main()