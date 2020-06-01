import os
import sys
import pickle
import numpy as np

bezier_curves = {
    1: np.array([[2, 1060], [400, 190]]),
    2: np.array([[990, 2], [150, 320]])
    
}



class BezierMatching:

    def __init__(self, cam_ident, bezier_curves=bezier_curves):
        #self.config = config['TRACKING']
        #self.default = config['DEFAULT']
        self.cam_ident = cam_ident
        self.bezier_curves = bezier_curves

        
        #self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'tracker_output', self.cam_ident) #set output directory
        #os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder
        print()

   
    ##
    # Determines parametric 't' for linear movements
    # @param coor Coordinate matrix, coor[0, i] is x-coor, coor[1, i] is y-coor
    # @param mvt Movement ID
    # @returns t parameter of bezier curve
    #
    def get_linear_t(self, coor, mvt):

        x1 = self.bezier_curves[mvt][0, 0]
        y1 = self.bezier_curves[mvt][1, 0]
        x2 = self.bezier_curves[mvt][0, 1]
        y2 = self.bezier_curves[mvt][1, 1]
        x_coor = coor[0, :]
        y_coor = coor[1, :]
        shape = np.shape(x_coor)

        x1_full = x1 * np.ones(shape)
        y1_full = y1 * np.ones(shape)

        t = -1 * ((x1_full - x_coor)* (x2 - x1) + (y1_full - y_coor) * (y2 - y1))
        t = t / (np.square(x2 - x1)  + np.square(y2 - y1))

        return t

    
    ##
    # Compares empirical coordinates to each movement
    # @param coor Coordinate matrix, coor[0, i] is x-coor, coor[1, i] is y-coor
    # @returns 
    #
    def project_on_movements(self, coor):

        for mvt in self.bezier_curves.keys():

            if np.shape(self.bezier_curves[mvt])[1] == 2: #linear movement
                t = self.get_linear_t(coor, mvt) 
            
            else:
                print()


    ##
    # Begins the bezier matching process.
    # @param data The pickle file stored from tracker.py 
    # @returns DefaultPredictor, cfg object
    #
    def proces_tracking_results(self, data):

        for trackerID in data.keys(): #for each tracked vehicle

            tracked_vehicle = data[trackerID]
            N = len(tracked_vehicle) #N boxes of tracked coordinates
            
            coor = np.zeros((2, N)) #accumulate coordinates

            for i in range(0, N):

                box = tracked_vehicle[i][1]
                # N columns, coor[0, i] is x-coor, coor[1, i] is y-coor
                coor[0,i] = box[0] + (box[2] - box[0]) / 2
                coor[1,i] = box[1] + (box[3] - box[1]) / 2

                  
            self.project_on_movements(coor) 




if __name__ == '__main__':
    print('hello world')

    with open(sys.argv[1], 'rb') as f:

        data = pickle.load(f)

    BezierMatching('cam_1').proces_tracking_results(data)