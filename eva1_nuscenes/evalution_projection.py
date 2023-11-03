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

def get_frame_from_video(video_name, frame_time, img_dir, img_name, keypoints, projectpoints):
    """
    get a specific frame of a video by time in milliseconds
    :param video_name: video name
    :param frame_time: time of the desired frame
    :param img_dir: path which use to store output image
    :param img_name: name of output image
    :return: None
    """
    vidcap = cv2.VideoCapture(video_name)
    # Current position of the video file in milliseconds.
    vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time - 1)
    # read(): Grabs, decodes and returns the next video frame
    success, image = vidcap.read()
    for keypoint in keypoints:
        cv2.circle(image, keypoint, 1, (255, 0, 0), 3)
    for projectpoint in projectpoints:
        cv2.circle(image, projectpoint, 5, (0, 0, 255), 5)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if success:
        # save frame as JPEG file
        cv2.imwrite(img_dir + img_name, image)
        # cv2.imshow("frame%s" % frame_time, image)
        # cv2.waitKey()



path_keypoints = './carkeypoints/'
path_mesh = './carmesh/'
path_rotation = './carrotation/'
path_translation = './cartranslation/'
filelist_keypoints = os.listdir(path_keypoints)
filelist_mesh = os.listdir(path_mesh)
filelist_rotation = os.listdir(path_rotation)
filelist_translation = os.listdir(path_translation)

# select frame 200.
keypoints = []
for file_keypoints in filelist_keypoints:
    frameid = int(re.findall('\d+', file_keypoints)[0])
    if frameid == 200:
        with open(path_keypoints + file_keypoints,'r') as f:
            for jf in f:
                eachdata = json.loads(jf)
                keypoints.append((int(eachdata[0]),int(eachdata[1])))
print(keypoints)

M = np.load('extrinsic.npy')
K = np.load('intrinsic.npy')
print(K)
K = np.array([[-1.25281310e+03, 0.00000000e+00, 8.26588115e+02],
    [0.00000000e+00, -1.25281310e+03, 4.69984663e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
projectpoints = []
for file_mesh in filelist_mesh:
    frameid = int(re.findall('\d+', file_mesh)[0])
    carid = int(re.findall('\d+', file_mesh)[1])
    if frameid == 200:
        mesh = loadMesh(path_mesh + file_mesh)
        mesh = np.array(mesh)
        center_x = (max(mesh[:,0]) + min(mesh[:,0]))/2
        center_y = (max(mesh[:,1]) + min(mesh[:,1]))/2
        center_z = (max(mesh[:,2]) + min(mesh[:,2]))/2
        center = np.array([[center_x],[center_y],[center_z]])
        file_tr = str('%04d' % frameid) + '_' + str('%04d' % carid) + '.json'
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
get_frame_from_video(video_name='video.mp4',frame_time=(200*1000/12.0),img_dir='./',img_name='img.png',keypoints=keypoints,projectpoints=projectpoints)