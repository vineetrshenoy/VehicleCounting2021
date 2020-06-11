import os
import sys
import numpy as np




class Helper():


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
            
            arr = np.zeros((int(M/2), 2))

            for j in range(0, int(M/2)):

                arr[j, 0] = int(coor_list[2*j])
                arr[j, 1] = int(coor_list[2*j + 1])

            mvt_dict[i +1] = arr.T

        return mvt_dict


if __name__ == '__main__':

    #Helper.get_roi('2,228,1916,300,1916,1076,2,1074')
    Helper.load_bezier_curve('/vulcan/scratch/vshenoy/vehicle_counting/src/bezier_curves/cam_9.txt')