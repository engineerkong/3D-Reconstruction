import numpy as np
import json
import os

def loadData(fileName):
    data = []
    with open(fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata)
    return data

def writeData(fileName, data):
    with open(fileName, 'w') as f:
        i = 0
        for d in data:
            i += 1
            # d = d.tolist()
            json.dump(d, f)
            if i != len(data):
                f.write('\n')

def pca(X):
    means = np.mean(X[:,:2],axis=0)
    Y = np.zeros((len(X),2))
    Y[:,:2] = X[:,:2] - means
    covM = np.dot(Y.T, Y)/len(Y)
    eigval, eigvec = np.linalg.eig(covM)
    W = eigvec
    return W

def findorientation(path_in_peoplekeypoints,path_out_people):
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)
    filelist_keypoints = os.listdir(path_in_peoplekeypoints)

    for file_keypoints in filelist_keypoints:
        keypoints = loadData(path_in_peoplekeypoints + file_keypoints)
        if keypoints != []:
            keypoints = np.asarray(keypoints)
            keypoints_vec = pca(keypoints)
            keypoints_lw = np.dot(keypoints[:, :2], keypoints_vec)
            l = abs(np.max(keypoints_lw[:,0]) - np.min(keypoints_lw[:,0]))
            w = abs(np.max(keypoints_lw[:,1]) - np.min(keypoints_lw[:,1]))
            h = abs(np.max(keypoints[:,2]) - np.min(keypoints[:,2]))
            yaw = np.arctan2(keypoints_vec[1, 0], keypoints_vec[0, 0])
            data = [l,w,h,yaw]
        else:
            data = [0,0,0,0]
        writeData(path_out_people + file_keypoints,data)
