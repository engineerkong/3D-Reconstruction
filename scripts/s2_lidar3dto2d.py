import cv2
import numpy as np
import os
from plyfile import PlyData
import pandas as pd
import re
import json

# K4 = [[1382.3642378, 0.0, 971.170879], [0.0, 1382.3642378, 540.91932], [0.0, 0.0, 1.0]]
# K5 = [[1372.8473736, 0.0, 957.43015], [0.0, 1372.8473736, 548.045805], [0.0, 0.0, 1.0]]
# R4 = [[-0.4554498804, 0.3337000564, -0.8253542747], [0.8902121339, 0.1804643161, -0.4182762094], [0.0093682001, -0.9252442396, -0.3792562905]]
# R5 = [[-0.4107574765, 0.3359876441, -0.847579258], [0.9113619105, 0.178238182, -0.3710129627], [0.0264152147, -0.9248478002, -0.3794190071]]
# T4 = [47.27359999995679, 16.252399999648333, 15.581100000000006]
# T5 = [47.39269999996759, 15.870600000023842, 15.602999999999994]
# R4_inv = np.array(R4).T

# the files in pah_in is from given lidar 3d points cloud
# the files in path_out will be the lidar 2d points in camera prospective
def lidar3dto2d(path_in,path_out):
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    K = [[846.2772801361476, 0.0, 988.4546133916215, 0.0], [0.0, 851.9068507035013, 533.3470761399655, 0.0], [0.0, 0.0, 1.0, 0.0]]
    M = np.zeros((4, 4))
    M[:3, :3] = cv2.Rodrigues(np.asarray([0.3442905614582061, 2.466378657279374, -1.7432149451377668]))[0]
    M[:3, 3] = [-17.1982144948334, 2.6602985635296443, 47.41931873523568]

    filelist = os.listdir(path_in)
    for file in filelist:
        number = re.findall('\d+', file)
        plys = []
        with open(path_in + file, 'r') as f:
            for jf in f:
                eachdata = json.loads(jf)
                plys.append(eachdata)
        index = 0
        for points in plys:
            threedpoints = np.array([points[0],points[1],points[2],1]).T
            threedpoints = np.dot(M,threedpoints)
            twodpoints = np.dot(K,threedpoints)
            scale = twodpoints[2]
            twodpoints = [twodpoints[0]/scale,twodpoints[1]/scale,index]
            index += 1
            if scale >= 0 and twodpoints[0] <= 1920 and twodpoints[0] >= 0 and twodpoints[1] <= 1080 and twodpoints[1] >= 0:
                with open(path_out + str(number[0]) + '.json', 'a') as f:
                    json.dump(twodpoints, f)
                    f.write('\n')
