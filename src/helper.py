import os
import sys
import numpy as np




class Helper():

    ##
    #   Processes the ROI from the config file in a format for code
    #   @param coor ROI string, stored as x1, y1, x2, y2,  ... , xn, yn
    #   @returns arr numpy-array of coordinates
    #
    @staticmethod
    def get_roi(coor:str):

        #coor = '2,228,1916,300,1916,1076,2,1074'
        coor_list = coor.split(',')
        N = len(coor_list)
        
        arr = np.zeros((int(N/2), 2))

        for i in range(0, int(N/2)):

            arr[i, 0] = int(coor_list[2*i])
            arr[i, 1] = int(coor_list[2*i + 1])

        return arr

    ##
    #   Loads the bezier curves from a file
    #   @param filename The file storing the bezier curves
    #   @returns mvt_dict Movements stored as a dictionary
    #
    @staticmethod
    def load_bezier_curve(filename):

        with open(filename, 'rb') as f:
            coor = f.readlines()

        N = len(coor)

        mvt_dict = {}

        for i in range(0, N):

            coor_list = coor[i].decode('utf-8').rstrip()

            coor_list = coor_list.split(',')
            M = len(coor_list)
            
            arr = np.zeros((2, int(M/2)))

            for j in range(0, int(M/2)):

                arr[0, j] = int(coor_list[2*j])
                arr[1, j] = int(coor_list[2*j + 1])

            mvt_dict[i +1] = arr

        return mvt_dict

    ##
    #   Loads the display locations from a file
    #   @param filename The file from which to get the locations
    #   @returns loc_dict The location dictionary
    #
    @staticmethod
    def load_display_locations(filename):

        with open(filename, 'rb') as f:
            coor = f.readlines()

        N = len(coor)

        loc_dict = {}

        for i in range(0, N):
            
            coor_list = coor[i].decode('utf-8').rstrip()
            coor_list = coor_list.split(',')

            loc_dict[i + 1] = (int(coor_list[0]), int(coor_list[1]))

        
        return loc_dict

if __name__ == '__main__':

    #Helper.get_roi('2,228,1916,300,1916,1076,2,1074')
    Helper.load_bezier_curve('src/bezier_curves/cam_6_snow.txt')