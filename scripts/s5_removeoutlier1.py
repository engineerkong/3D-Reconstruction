import json
import numpy as np
import os

def loadData(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            eachdata = [eachdata[0], eachdata[1], eachdata[2]]
            data.append(eachdata)
        data=np.array(data)
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

def outlier(data):
    x = data
    print(np.shape(x)[0])
    outliers = []

    for i in range(np.shape(x)[1]):
        temp = []
        for j in range(np.shape(x)[0]):
            temp.append(x[j][i])
        Q1, Q3 = np.percentile(temp, [30, 70])

        iqr = Q3 - Q1
        min = Q1 - (0.75 * iqr)
        max = Q3 + (0.75 * iqr)
        for j in range(0, np.shape(x)[0]):
            if (x[j][i] < min or x[j][i] > max):
                outliers.append(j)

        x_outliers = np.delete(x, outliers, axis=0)

    print(np.shape(x_outliers)[0])
    return x_outliers

def removeoutlier1(path_in_car,path_in_people,path_out_car,path_out_people):
    if not os.path.exists(path_out_car):
        os.makedirs(path_out_car)
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)
    filelist_car = os.listdir(path_in_car)
    filelist_people = os.listdir(path_in_people)
    for file_car in filelist_car:
        print(file_car)
        inputdata = loadData(path_in_car + file_car)
        if inputdata != []:
            outputdata = outlier(data=inputdata)
        else:
            outputdata = inputdata
        outputData(path_out_car + file_car,outputdata)
    for file_people in filelist_people:
        print(file_people)
        inputdata = loadData(path_in_people + file_people)
        if inputdata != []:
            outputdata = outlier(data=inputdata)
        else:
            outputdata = inputdata
        outputData(path_out_people + file_people,outputdata)