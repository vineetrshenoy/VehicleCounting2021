import os
import sys
import pickle
import numpy as np
import configparser
import glob
from tqdm import tqdm

config = configparser.ConfigParser()
config.read(sys.argv[1])

bezier_curves = {
    
    1: np.array([[990, 2], [150, 320]]),
    2: np.array([[2, 1060], [400, 190]]),
    3: np.array([[1275, -50, 1060], [460, 400, 190]]),
    4: np.array([[2, 600, 1275], [500, 450, 545]])
}



class BezierMatching:

    def __init__(self, cam_ident, bezier_curves=bezier_curves):
        self.config = config['BEZIER']
        self.default = config['DEFAULT']
        self.cam_ident = cam_ident
        self.bezier_curves = bezier_curves
        
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'counting_output') #set output directory
        os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder
        self.track1txt = open(os.path.join(self.out_dir, self.default['job_name'] + '.txt'), 'w')


    
    ##
    # Get point on curve such that tracker point to curve and curve itself are orthogonal
    # @param t parameter for points on curve
    # @param mvt Movement ID
    # @param coor tracker points
    # @returns curve_points Points on curve orthogonal to tracker points
    #
    def get_linear_score(self, t, mvt, coor):

        N  = len(t)
        points = self.bezier_curves[mvt]

        #control points p1, p2
        p1 = points[:, 0].reshape(1,2)
        p2 = points[:, 1].reshape(1,2)

        #All points for all values of t
        endpoint_diff = np.tile((p2 - p1), (N, 1)) #repeat (p2 -p1) for vectorized calc
        curve_points = np.tile(p1, (N, 1)) + t.reshape(N,1) * endpoint_diff #points on curve
        point_curve_diff = curve_points - coor.T #diff between tracker point and point on curve
       
        dot_product = np.sum(point_curve_diff*endpoint_diff, axis=1)       
        
        assert np.sum(dot_product) < 1e-5 #assert orthogonal property is satisfied
        
        return np.linalg.norm(point_curve_diff)


    ##
    # Get point on curve such that tracker point to curve and curve itself are orthogonal
    # @param t parameter for points on curve
    # @param mvt Movement ID
    # @param coor tracker points
    # @returns curve_points Points on curve orthogonal to tracker points
    #
    def get_quadratic_score(self, t, mvt, coor):

        N  = len(t)
        points = self.bezier_curves[mvt]
        
        #control points pt1, pt2, pt3
        pt1 = points[:, 0].reshape(1,2) 
        pt2 = points[:, 1].reshape(1,2) 
        pt3 = points[:, 2].reshape(1,2)

        #All points for all values of t
        pts = np.kron(np.square(1 - t), pt1) + np.kron(2 * (1 - t) * t, pt2) + np.kron(np.square(t), pt3) #shape: (1, 2*N)
        pts = pts.reshape((N,2)) #rows of [x, y] coordinates
        diff = pts - coor.T
        dt = 2*(np.kron((t-1), pt1) + np.kron((1-2*t), pt2) + np.kron(t, pt3)) #shape: (1, 2*N)
        dt = dt.reshape((N,2)) #rows of [x, y] coordinates

        dot_product = np.sum(dt*diff, axis=1)      
        
        assert np.sum(dot_product) < 1e-5 #assert orthogonal property is satisfied
        
        return np.linalg.norm(diff)


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

        x1_full = x1 * np.ones(shape) #expand x1 for vectorized calculation
        y1_full = y1 * np.ones(shape) #expand y1 for vectorized calculation

        t = -1 * ((x1_full - x_coor)* (x2 - x1) + (y1_full - y_coor) * (y2 - y1))
        t = t / (np.square(x2 - x1)  + np.square(y2 - y1))

        return t

    
    ##
    # Determines parametric 't' for quadratic movements
    # @param coor Coordinate matrix, coor[0, i] is x-coor, coor[1, i] is y-coor
    # @param mvt Movement ID
    # @returns t parameter of bezier curve
    #
    def get_quadratic_t(self, coor, mvt):

        x1 = self.bezier_curves[mvt][0, 0]
        y1 = self.bezier_curves[mvt][1, 0]
        x2 = self.bezier_curves[mvt][0, 1]
        y2 = self.bezier_curves[mvt][1, 1]
        x3 = self.bezier_curves[mvt][0, 2]
        y3 = self.bezier_curves[mvt][1, 2]

        xn = coor[0, :]
        yn = coor[1, :]
        coor_len = len(xn)

        #coefficients for cubic equation solver
        fourth = (x1**2 + 4*x2**2 + x3**2 - 4*x1*x2 + 2*x1*x3 - 4*x2*x3) + (y1**2 + 4*y2**2 + y3**2 - 4*y1*y2 + 2*y1*y3 - 4*y2*y3)
        third = (-3*x1**2 - 6*x2**2 + 9*x1*x2 - 3*x1*x3 + 3*x2*x3) + (-3*y1**2 - 6*y2**2 + 9*y1*y2 - 3*y1*y3 + 3*y2*y3)
        second = (3*x1**2 + 2*x2**2 - 6*x1*x2 + x1*x3 - xn*x1 + 2*xn*x2 - xn*x3) + (3*y1**2 + 2*y2**2 - 6*y1*y2 + y1*y3 - yn*y1 + 2*yn*y2 - yn*y3)
        first = (-1*x1**2 + x1*x2 + xn*x1 - xn*x2) + (-1*y1**2 + y1*y2 + yn*y1 - yn*y2)

        t = np.zeros(coor_len)


        def get_shortest_distance_root_index(roots, indices, i):
            
            points = self.bezier_curves[mvt]
        
            #control points pt1, pt2, pt3
            pt1 = points[:, 0].reshape(1,2) 
            pt2 = points[:, 1].reshape(1,2) 
            pt3 = points[:, 2].reshape(1,2)
            N = len(indices)
            
            t_val = roots[indices]
            pts = np.kron(np.square(1 - t_val), pt1) + np.kron(2 * (1 - t_val) * t_val, pt2) + np.kron(np.square(t_val), pt3)
            pts = pts.reshape((N,2)) #rows of [x, y] coordinates
            distance = np.square(pts - np.tile(coor[:, i].T, (N, 1)))
            distance = np.sum(distance, axis=1)
            
            return t_val[np.argmin(distance)]

            
        for i in range(0, coor_len): #need loop because np.roots can not be vectorized
            roots = np.roots([fourth, third, second[i], first[i]]) #only 'second' and 'first' depend on different xn,yn coordinates
            indices = np.where(np.logical_and(roots>=0, roots<=1))[0]
            t[i] = get_shortest_distance_root_index(roots, indices, i)

        
        return t


    ##
    # Compares empirical coordinates to each movement
    # @param coor Coordinate matrix, coor[0, i] is x-coor, coor[1, i] is y-coor
    # @returns 
    #
    def project_on_movements(self, coor):

        mvt_scores = np.zeros(len(self.bezier_curves.keys()))
        for mvt in self.bezier_curves.keys():

            if np.shape(self.bezier_curves[mvt])[1] == 2: #linear movement
                t = self.get_linear_t(coor, mvt) 
                if np.sum(np.diff(t) < 0) / len(t) > float(self.config['THRESHOLD']): #t is decreasing -- wrong direction
                    mvt_scores[mvt - 1] = np.iinfo(np.int32).max
                else:
                    score = self.get_linear_score(t, mvt, coor)
                    mvt_scores[mvt - 1] = score
            
            else: #quadratic movement
                t = self.get_quadratic_t(coor, mvt)
                if np.sum(np.diff(t) < 0) / len(t) > float(self.config['THRESHOLD']): #t is decreasing -- wrong direction
                    mvt_scores[mvt - 1] = np.iinfo(np.int32).max
                else:
                    score = self.get_quadratic_score(t, mvt, coor)
                    mvt_scores[mvt - 1] = score
        
        return np.argmin(mvt_scores) + 1

    ##
    # Begins the bezier matching process.
    # @param data The pickle file stored from tracker.py 
    # @returns DefaultPredictor, cfg object
    #
    def process_tracking_results(self, data, cat_id):

        for trackerID in tqdm(sorted(data.keys())): #for each tracked vehicle

            tracked_vehicle = data[trackerID]
            N = len(tracked_vehicle) #N boxes of tracked coordinates
            
            coor = np.zeros((2, N)) #accumulate coordinates

            for i in range(0, N):

                box = tracked_vehicle[i][1]
                # N columns, coor[0, i] is x-coor, coor[1, i] is y-coor
                coor[0,i] = box[0] + (box[2] - box[0]) / 2
                coor[1,i] = box[1] + (box[3] - box[1]) / 2

                  
            mvt_id = self.project_on_movements(coor)
            frame_id = tracked_vehicle[N - 1][0] + 1 #tracker is zero-indexed
            video_id = 1

            self.track1txt.write('{} {} {} {} {}\n'.format(video_id, frame_id, mvt_id, cat_id, trackerID))

        

    ##
    # Begins the bezier matching process.
    # @param data The pickle file stored from tracker.py 
    # @returns DefaultPredictor, cfg object
    #
    def workflow(self):
        
        tracker_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'tracker_output', self.cam_ident)
        category_dict = {'Car': 1, 'Truck': 2, 'Bus': 2}


        files = glob.glob(os.path.join(tracker_dir, 'Track*'))

        for tracker_file in files:

            with open(tracker_file, 'rb') as f:              
                
                tracker_results = pickle.load(f)

                filename = os.path.basename(tracker_file)
                cat = filename.split('_')[1]
                cat_id = category_dict[cat]
                self.process_tracking_results(tracker_results, cat_id)

        self.track1txt.close()


    


if __name__ == '__main__':
    
   
    BezierMatching('cam_1').workflow()