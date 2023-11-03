import os
import json
import re
from plyfile import PlyData
import pandas as pd

# return the matched points back to lidar 3d
def get3dlidar(path_in_car,path_in_people,path_in_lidar,path_out_car,path_out_people):
    if not os.path.exists(path_out_car):
        os.makedirs(path_out_car)
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)
    filelist_car = os.listdir(path_in_car)
    filelist_people = os.listdir(path_in_people)
    filelist_lidar = os.listdir(path_in_lidar)

    for file_lidar in filelist_lidar:
        lidarid = re.findall('\d+', file_lidar)
        plys = []
        with open(path_in_lidar + file_lidar, 'r') as f:
            for jf in f:
                eachdata = json.loads(jf)
                plys.append(eachdata)
        for file_car in filelist_car:
            if str('%04d' % int(lidarid[0])) + '_' in file_car:
                points_index = []
                with open(path_in_car + file_car, 'r') as f:
                    for jf in f:
                        eachdata = json.loads(jf)
                        points_index.append(int(eachdata[2]))
                carpoints = []
                index = 0
                for plypoint in plys:
                    if index in points_index:
                        carpoints.append(plypoint)
                    index += 1
                with open(path_out_car + file_car, 'w') as f:
                    i = 0
                    for point in carpoints:
                        i += 1
                        json.dump(point, f)
                        if i != len(carpoints):
                            f.write('\n')
        for file_people in filelist_people:
            if str('%04d' % int(lidarid[0])) + '_' in file_people:
                points_index = []
                with open(path_in_people + file_people, 'r') as f:
                    for jf in f:
                        eachdata = json.loads(jf)
                        points_index.append(int(eachdata[2]))
                peoplepoints = []
                index = 0
                for plypoint in plys:
                    if index in points_index:
                        peoplepoints.append(plypoint)
                    index += 1
                with open(path_out_people + file_people, 'w') as f:
                    i = 0
                    for point in peoplepoints:
                        i += 1
                        json.dump(point, f)
                        if i != len(peoplepoints):
                            f.write('\n')