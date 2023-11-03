import numpy as np
import os
import re
import json
from kd_tree import *

def loadData1(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            eachdata = [eachdata[0], eachdata[1]]
            data.append(eachdata)
    return data

def loadData2(fileName):
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
            d = d.tolist()
            json.dump(d, f)
            if i != len(data):
                f.write('\n')

def matching_function(src,dst,dst3info):
    A = np.array(src)
    B = np.array(dst3info)
    kdtree = KDTree(dst,2)
    kdtree_dis = []
    kdtree_points = []
    for t in A:
        kdtree_dis.append(kdtree.get_nearest(t)[0])
        kdtree_points.append(kdtree.get_nearest(t)[1])
    kdtree_dis = np.array(kdtree_dis)
    kdtree_points = np.array(kdtree_points)
    dis = np.mean(kdtree_dis)
    print(dis)
    C = np.zeros((len(src),3))
    for i in range(len(src)):
        kdtree_point = kdtree_points[i,:]
        for point in B:
            if kdtree_point[0]==point[0] and kdtree_point[1]==point[1]:
                index = point[2]
        C[i,0] = A[i,0]
        C[i,1] = A[i,1]
        C[i,2] = index
    return C

def matching2d(path_in_people,path_in_carkeypoints,path_in_lidar2d,ratio,path_out_people,path_out_car):
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)
    if not os.path.exists(path_out_car):
        os.makedirs(path_out_car)
    with open(path_in_people) as f:
        people_list = json.load(f)
    filelist_carkeypoints = os.listdir(path_in_carkeypoints)
    filelist_lidar2d = os.listdir(path_in_lidar2d)

    for file_lidar2d in filelist_lidar2d:
        lidarid = re.findall('\d+', file_lidar2d)
        lidarid = int(lidarid[0])
        print(lidarid)
        frameid = lidarid * ratio + 1
        data_lidar = loadData1(path_in_lidar2d + file_lidar2d)
        data_lidar_index = loadData2(path_in_lidar2d + file_lidar2d)
        for people_dict in people_list:
            imageid = people_dict['image_id']
            imageid = re.findall('\d+', imageid)
            imageid = int(imageid[0])
            imageid = imageid + 1
            if imageid == frameid:
                index = people_dict['idx']
                keypoints = people_dict['keypoints']
                data_people = []
                for i in range(17):  # 17 skeleton keypoints on one people
                    data_people.append([keypoints[i * 3], keypoints[i * 3 + 1]])
                file_out = str('%04d' % lidarid) + '_' + str('%04d' % index) + '.json'
                file_out = path_out_people + file_out
                data_out = matching_function(data_people, data_lidar, data_lidar_index)
                outputData(file_out,data_out)
        for file_carkeypoints in filelist_carkeypoints:
            carname = re.findall('\d+', file_carkeypoints)
            imageid = int(carname[0])
            if imageid == frameid:
                index = int(carname[1])
                data_car = loadData1(path_in_carkeypoints + file_carkeypoints)
                file_out = str('%04d' % lidarid) + '_' + str('%04d' % index) + '.json'
                file_out = path_out_car + file_out
                data_out = matching_function(data_car, data_lidar, data_lidar_index)
                outputData(file_out,data_out)