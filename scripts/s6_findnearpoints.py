import numpy as np
import os
import json
import re
from plyfile import PlyData
import pandas as pd
from kd_tree import *

def loadData(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata)
    return data

def outputData(fileName, data):
    with open(fileName, 'w') as f:
        i = 0
        for d in data:
            i += 1
            json.dump(d, f)
            if i != len(data):
                f.write('\n')

def nearpoints(data1, data2, dis):
    data3 = []
    for d1 in data1:
        data3.append(d1)
        d1 = np.array(d1)
        for d2 in data2:
            d2 = np.array(d2)
            distance = np.linalg.norm((d1-d2))
            if distance < dis:
                d2 = d2.tolist()
                data3.append(d2)
    result = []
    for item in data3:
        if item not in result:
            result.append(item)
    return result

# find the near points around of cara and people
def findnearpoints(path_in_car,path_in_people,path_in_lidar,path_out_car,path_out_people):
    if not os.path.exists(path_out_car):
        os.makedirs(path_out_car)
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)

    filelist_car = os.listdir(path_in_car)
    filelist_people = os.listdir(path_in_people)
    filelist_lidar = os.listdir(path_in_lidar)
    n_car = 600
    d_car = 2.5 # center to edge and the distance with other car
    n_people = 150
    d_people = 1.5 # center to foot or head and the distance with other people
    for file_lidar in filelist_lidar:
        lidarid = re.findall('\d+', file_lidar)
        print(lidarid)
        plys = []
        with open(path_in_lidar + file_lidar, 'r') as f:
            for jf in f:
                eachdata = json.loads(jf)
                plys.append(eachdata)
        kdtree = KDTree(plys, 3)
        for file_car in filelist_car:
            if str('%04d' % int(lidarid[0])) + '_' in file_car:
                carpoints = loadData(path_in_car + file_car)
                result = []
                for t in carpoints:
                    kdtree_points = kdtree.get_dnn(t,n_car,d_car)
                    for point in kdtree_points:
                        if point not in result:
                            result.append(point)
                outputData(path_out_car + file_car, result)
        for file_people in filelist_people:
            if str('%04d' % int(lidarid[0])) + '_' in file_people:
                peoplepoints = loadData(path_in_people + file_people)
                result = []
                for t in peoplepoints:
                    kdtree_points = kdtree.get_dnn(t,n_people,d_people)
                    for point in kdtree_points:
                        if point not in result:
                            result.append(point)
                outputData(path_out_people + file_people, result)
