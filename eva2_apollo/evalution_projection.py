import numpy as np
import json
import os
import cv2
import re
import math

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


def loadDict(fileName):
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            for value in eachdata.values():
                data = value
    return data

def RPYAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(np.dot(R_z, R_y), R_x)
    return R

image = cv2.imread('apollo.jpg')

path_keypoints = './carkeypoints/'
path_mesh = './carmesh/'
path_rotation = './carrotation/'
path_translation = './cartranslation/'
filelist_keypoints = os.listdir(path_keypoints)
filelist_mesh = os.listdir(path_mesh)
filelist_rotation = os.listdir(path_rotation)
filelist_translation = os.listdir(path_translation)

K = np.load('camera_intrinsic.npy')
print(K)

keypoints = []
for file_keypoints in filelist_keypoints:
    with open(path_keypoints + file_keypoints,'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            keypoints.append((int(eachdata[0]),int(eachdata[1])))
print(keypoints)

projectpoints = []
for file_mesh in filelist_mesh:
    id = frameid = int(re.findall('\d+', file_mesh)[0])
    mesh = loadMesh(path_mesh + file_mesh)
    mesh = np.array(mesh)
    center_x = (max(mesh[:,0]) + min(mesh[:,0]))/2
    center_y = (max(mesh[:,1]) + min(mesh[:,1]))/2
    center_z = (max(mesh[:,2]) + min(mesh[:,2]))/2
    center = np.array([[center_x],[center_y],[center_z]])
    file_tr = str(id) + '.json'
    rotation = loadDict(path_rotation + file_tr)
    R = RPYAnglesToRotationMatrix(rotation)
    translation = loadDict(path_translation + file_tr)
    T = np.array([[translation[0]], [translation[1]], [translation[2]]])
    car_cam = np.dot(R, center) + T
    print(car_cam)
    car_img = np.dot(K, car_cam)
    car_img = car_img.tolist()
    car_img = (int(car_img[0][0]/car_img[2][0]), int(car_img[1][0]/car_img[2][0]))
    projectpoints.append(car_img)
print(projectpoints)

for keypoint in keypoints:
    cv2.circle(image, keypoint, 3, (0, 255, 0), 3)
for projectpoint in projectpoints:
    cv2.circle(image, projectpoint, 8, (0, 0, 255), 12)
cv2.imwrite('img.jpg', image)