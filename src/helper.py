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


if __name__ == '__main__':

    Helper.get_roi('2,228,1916,300,1916,1076,2,1074')