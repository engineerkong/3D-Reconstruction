import numpy as np
import json
import os
import random
from plyfile import PlyData
import pandas as pd
import re
import open3d

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def ransac(alldata, threshold, estimate, is_inlier, goal_inliers, stop_at_goal=True, random_seed=None):
    best_ic = 0
    max_iterations = 20
    sample_size = 10
    best_model = None
    random.seed(random_seed)
    for i in range(max_iterations):
        s = random.sample(alldata, int(sample_size))
        m = estimate(s)
        ic = 0
        data_rm = []
        for j in range(len(alldata)):
            if is_inlier(m, alldata[j],threshold):
                ic += 1
                eachdata = alldata[j]
                data_rm.append(eachdata)
        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_data_rm = data_rm
            if ic > goal_inliers and stop_at_goal:
                break
        print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic)
    return best_data_rm

# the files in path_in_car and path_in_people are from step6 after getting lidar 3d points
# the files in path_in_nearcar and path_in_nearpeople are from step7 after finding nearpoints
# the files in path_out_car and path_out_people will go to next step
def planefitting(path_in_lidar,path_out):
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    filelist_lidar = os.listdir(path_in_lidar)
    for file_lidar in filelist_lidar:
        tmp = os.path.join(path_in_lidar, file_lidar)
        plydata = PlyData.read(tmp)
        ply = plydata['vertex']
        plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
        plys = plys.tolist()
        n = len(plys)
        print('plysnumber:' + str(len(plys)))
        name_lidar = re.findall('\d+', file_lidar)
        print('lidar:' + str('%04d' % int(name_lidar[0])))
        goal_inliers = n * 0.3
        threshold = 0.1
        print('planefitting start')
        plys_rm = ransac(plys, threshold, estimate, is_inlier, goal_inliers)
        print('planefitting finished')
        print('plysnumber removing:' + str(len(plys_rm)))
        plys_out = plys
        index = 0
        for d in plys_rm:
            index += 1
            if index % 10000 == 0:
                print('removed:' + str(index))
            plys_out.remove(d)
        # plys_out = []
        # index = 0
        # for d in plys:
        #     if d not in plys_rm:
        #         plys_out.append([d[0],d[1],d[2],index])
        #     index += 1
        print('plysnumber left:' + str(len(plys_out)))
        with open(path_out + str('%04d' % int(name_lidar[0])) + '.json', 'w') as f:
            i = 0
            for d in plys_out:
                i += 1
                json.dump(d, f)
                if i != len(plys_out):
                    f.write('\n')


