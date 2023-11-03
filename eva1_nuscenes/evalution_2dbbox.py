import os
import re
import numpy as np
import json

def loadDict(fileName):
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            for value in eachdata.values():
                data = value
    return data

def image_box_overlap(boxes, query_boxes, threshold, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qb_x_min = query_boxes[k,0]
        qb_y_min = query_boxes[k,1]
        qb_x_max = query_boxes[k,0] + query_boxes[k,2]
        qb_y_max = query_boxes[k,1] + query_boxes[k,3]
        qbox_area = ((qb_x_max - qb_x_min) *
                     (qb_y_max - qb_y_min))
        for n in range(N):
            b_x_min = boxes[n, 0]
            b_y_min = boxes[n, 1]
            b_x_max = boxes[n, 0] + boxes[n, 2]
            b_y_max = boxes[n, 1] + boxes[n, 3]
            iw = (min(b_x_max, qb_x_max) -
                  max(b_x_min, qb_x_min))
            if iw > 0:
                ih = (min(b_y_max, qb_y_max) -
                      max(b_y_min, qb_y_min))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (b_x_max - b_x_min) *
                            (b_y_max - b_y_min) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((b_x_max - b_x_min) *
                            (b_y_max - b_y_min))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    iou = iw * ih / ua
                    if iou >= threshold:
                        iou = 1
                    else:
                        iou = 0
                    overlaps[n, k] = iou
    return overlaps

# define path
file_gt = 'mot.txt'
path_car = './carboxes/'
filelist_car = os.listdir(path_car)
file_people = 'alphapose_results_nuscenes.json'
with open(file_people) as f:
    people_list = json.load(f)

# load ground truth
old_detectionid = -1
txt_result = []
with open(file_gt, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').split(',')
        maxindex = int(list(line)[0])
        detectionid = int(list(line)[1])
        if detectionid != old_detectionid:
            txt_result.append(list(line))
        old_detectionid = detectionid
boxes_gt = []
for index in range(maxindex + 1):
    boxes_gt_index = []
    for d in txt_result:
        if int(d[0]) == index:
            boxes_gt_index.append([float(d[2]),float(d[3]),float(d[4]),float(d[5])])
    boxes_gt.append(boxes_gt_index)
print('boexes of ground truth:' + str(boxes_gt))

# load results from nets
boxes_net = []
for index in range(maxindex + 1):
    boxes_net_index = []
    for people_dict in people_list:
        image_id = people_dict['image_id']
        image_id = re.findall('\d+', image_id)
        image_id = int(image_id[0])
        if image_id == index:
            peopleboxes = people_dict['box']
            boxes_net_index.append([peopleboxes[0],peopleboxes[1],peopleboxes[2],peopleboxes[3]]) # tlx, tly, w, h
    for file_car in filelist_car:
        frame_id = re.findall('\d+', file_car)
        frame_id = int(frame_id[0])
        if frame_id == index + 1:
            carboxes = loadDict(path_car + file_car)
            boxes_net_index.append([carboxes[0],carboxes[1],carboxes[2] - carboxes[0],carboxes[3] - carboxes[1]]) # tlx, tly, w, h
    boxes_net.append(boxes_net_index)
print('boxes from net:' + str(boxes_net))

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative
for index in range(maxindex + 1):
    gt = np.asarray(boxes_gt[index])
    net = np.asarray(boxes_net[index])
    overlaps1 = image_box_overlap(net,gt,0.35)
    for overlap in overlaps1:
        if 1 in overlap:
            tp += 1
        else:
            fp += 1
    overlaps2 = image_box_overlap(gt,net,0.35)
    for overlap in overlaps2:
        if 1 not in overlap:
            fn += 1
print('true positive:' + str(tp))
print('false positive:' + str(fp))
print('false negative:' + str(fn))
