import numpy as np
import os
import json
import re
import csv

import pandas as pd


def loadData(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata)
    return data

def loadData2(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            outputdata = [eachdata[0],eachdata[1]]
            data.append(outputdata)
    return data

def loadDict(fileName):
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            for value in eachdata.values():
                data = value
    return data

def loadMesh(fileName):
    data = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            if len(tmp) == 4:
                if tmp[0] == 'v':
                    vertices = np.array([float(tmp[1]), float(tmp[2]), float(tmp[3])])
                    vertices = 2 * vertices
                    data.append(vertices)
    return data

# def pca(X):
#     means = np.mean(X[:,:2],axis=0)
#     Y = np.zeros((len(X),2))
#     Y[:,:2] = X[:,:2] - means
#     covM = np.dot(Y.T, Y)/len(Y)
#     eigval, eigvec = np.linalg.eig(covM)
#     W = eigvec
#     return W
#
# def outlier(data, percent):
#     x = data
#     numOutliers = 0
#     outliers = []
#
#     for i in range(np.shape(x)[1]):
#         temp = []
#         for j in range(np.shape(x)[0]):
#             temp.append(x[j][i])
#         Q1, Q3 = np.percentile(temp, [percent, 100-percent])
#
#         iqr = Q3 - Q1
#         min = Q1 - (1.0 * iqr)
#         max = Q3 + (1.0 * iqr)
#         for j in range(0, np.shape(x)[0]):
#             if (x[j][i] < min or x[j][i] > max):
#                 numOutliers += 1
#                 outliers.append(j)
#
#         x_outliers = np.delete(x, outliers, axis=0)
#
#     return x_outliers

def outputinfo(ratio, path_in_lidar, path_in_carresult, path_in_carkeypoints, path_in_carbbox, path_in_carscore, path_in_carmesh,
               path_in_translation, path_in_rotation, path_in_peopleresult, path_in_peoplelwhyaw, path_in_peopledir, path_out):
    global peopleh, peoplel, peoplew, peopley, peoplez, peoplex
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    filelist_lidar = os.listdir(path_in_lidar)
    filelist_lidar.sort()
    filelist_carresult = os.listdir(path_in_carresult)
    filelist_carresult.sort()
    filelist_peopleresult = os.listdir(path_in_peopleresult)
    filelist_peopleresult.sort()
    filelist_translation = os.listdir(path_in_translation)
    filelist_rotation = os.listdir(path_in_rotation)

    list = []
    with open(path_in_peopledir) as f:
        people_list = json.load(f)
    for file_lidar in filelist_lidar:
        # LiDAR Frame
        lidarid = int(re.findall('\d+', file_lidar)[0])
        print('LiDAR Frame:' + str(lidarid))
        for file_car in filelist_carresult:
            if str('%04d' % lidarid) + '_' in file_car:
                # Camera Frame
                frameid = int(re.findall('\d+', file_car)[0]) * ratio + 1
                # Detection ID
                carid = int(re.findall('\d+', file_car)[1])
                # filename from GSNet mesh
                file_car_mesh = str('%04d' % frameid) + '_' + str('%04d' % carid) + '.obj'
                # filename from GSNet
                file_car_net = str('%04d' % frameid) + '_' + str('%04d' % carid) + '.json'
                # filename for trans and yaw
                file_car_tr = str('%04d' % carid) + '.json'
                # 2DBBox tl.x tl.y bbwidth bbheigth
                carbbox_fromfile = loadDict(path_in_carbbox + file_car_net)
                xmin = carbbox_fromfile[0]
                ymin = carbbox_fromfile[1]
                xmax = carbbox_fromfile[2]
                ymax = carbbox_fromfile[3]
                carbbox = [xmin,ymin,(xmax-xmin),(ymax-ymin)]
                # Confidence
                carscore = loadDict(path_in_carscore + file_car_net)
                # Label car = 1
                # Visibility default = 1
                # 3DBBox LWH
                carmesh = loadMesh(path_in_carmesh + file_car_mesh)
                carmesh = np.array(carmesh)
                carl = max(carmesh[:,2]) - min(carmesh[:,2])
                carw = max(carmesh[:,0]) - min(carmesh[:,0])
                carh = max(carmesh[:,1]) - min(carmesh[:,1])
                carlwh = [carl, carw, carh]
                # 3DBBox Position
                if file_car_tr in filelist_translation:
                    translation = loadData(path_in_translation + file_car_tr)
                    if translation != []:
                        carposition = [translation[0],translation[1],translation[2]]
                    else:
                        carposition = [0,0,0]
                        carlwh = [0,0,0]
                else:
                    carposition = [0,0,0]
                    carlwh = [0,0,0]
                # 3DBBox Yaw
                if file_car_tr in filelist_rotation:
                    a = loadData(path_in_rotation + file_car_tr)
                    if a != []:
                        yaw = a
                    else:
                        yaw = 0
                else:
                    yaw = 0
                # 2D Keypoints
                carkeypoints = loadData2(path_in_carkeypoints + file_car_net)
                # 3D Model
                model3d = str('%04d' % frameid) + '_' +  str('%04d' % carid) + '.obj'
                # output
                list.append([lidarid,carid,carbbox,carscore,'1','1',carposition,carlwh,yaw,carkeypoints,frameid,model3d])
        print('car end')
        for file_people in filelist_peopleresult:
            if str('%04d' % lidarid) + '_' in file_people:
                # Camera Frame
                frameid = int(re.findall('\d+', file_people)[0]) * ratio + 1
                # Detection ID
                peopleid = int(re.findall('\d+', file_people)[1])
                for people_dict in people_list:
                    image_id = people_dict['image_id']
                    image_id = re.findall('\d+', image_id)
                    image_id = int(image_id[0])
                    if image_id == (frameid -1) and people_dict['idx'] == peopleid:
                        # 2DBBox tl.x tl.y bbwidth bbheigth
                        peoplebbox = people_dict['box']
                        peoplebbox = [peoplebbox[0],peoplebbox[1],peoplebbox[2],peoplebbox[3]]
                        # Confidence
                        peoplescore = people_dict['score']
                        # Label people = 2
                        # Visibility default = 1
                        # 3DBBox Position
                        peopleresult = loadData(path_in_peopleresult + file_people)
                        if peopleresult != []:
                            peopleresult = np.array(peopleresult)
                            peoplex = (np.max(peopleresult[:,0]) + np.min(peopleresult[:,0])) / 2
                            peopley = (np.max(peopleresult[:,1]) + np.min(peopleresult[:,1])) / 2
                            peoplez = np.max(peopleresult[:,2])
                            peopleposition = [peoplex, peopley, peoplez]
                        else:
                            peopleposition = [0,0,0]
                        # 3DBBox LWH Yaw
                        peoplelwhyaw = loadData(path_in_peoplelwhyaw + file_people)
                        peoplelwh = [peoplelwhyaw[0], peoplelwhyaw[1], peoplelwhyaw[2]]
                        yaw = peoplelwhyaw[3]
                        # 2D Keypoints
                        pkeypoints = people_dict['keypoints']
                        peoplekeypoints = []
                        for i in range(17):
                            peoplekeypoints.append([pkeypoints[3*i],pkeypoints[3*i+1]])
                        # no 3D Model for people
                        model3d = 'None'
                        # output
                        list.append([lidarid, peopleid, peoplebbox, peoplescore, '2', '1', peopleposition, peoplelwh, yaw, peoplekeypoints,
                                    frameid, model3d])
        print('people end')
    name = ['LiDAR Frame','Detection ID','2DBBox','Confidence','Label','Visibility',
            '3DBBox Position','3DBBox LWH','3DBBox Yaw','2DKeypoints','Camera Frame','3D Model']
    test = pd.DataFrame(columns=name,data=list)
    print(test)
    test.to_csv(path_out + 'output.csv',encoding='gbk')
if __name__ == "__main__":
    outputinfo(ratio=3,path_in_lidar='../input/lidarcloud/',path_in_carresult='../result_new/res7car/', path_in_carkeypoints='../input/carkeypoints/',
               path_in_carbbox='../input/carbbox/', path_in_carscore='../input/carscore/',path_in_carmesh='../input/carmesh/',
               path_in_translation='../output_new/car_translation/', path_in_rotation='../output_new/car_yaw/', path_in_peopleresult='../result_new/res7people/',
               path_in_peoplelwhyaw='../output_new/people_lwhyaw/',path_in_peopledir='../input/peoplekeypoints.json', path_out='../output_new/')
