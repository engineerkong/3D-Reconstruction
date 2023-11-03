import skvideo.io
import cv2
import csv
import json

def loadCsv(fileName):
    file = open(fileName)
    csvreader = csv.reader(file)
    # header = next(csvreader)
    # print(header)
    rows = []
    i = 0
    for row in csvreader:
        if i != 0:
            rows.append(row)
        i = 1
    file.close()
    return rows

def draw_bbox(img, labelbbox):
    label = labelbbox[0]
    top_left = (int(labelbbox[1]),int(labelbbox[2]))
    bottom_right = (int(labelbbox[1]+labelbbox[3]),int(labelbbox[2]+labelbbox[4]))
    if label == 1:
        cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0), thickness=2)
    else:
        cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=2)
    return img


def draw_car(img,labelkeypoints):
    for i in range(len(labelkeypoints[1])):
        keypoint = (int(labelkeypoints[1][i][0]),int(labelkeypoints[1][i][1]))
        cv2.circle(img, keypoint, 1, (0, 0, 255), 1)
    return img

def draw_people(img,labelkeypoints):
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    part_line = {}
    print(len(labelkeypoints[1]))
    for n in range(len(labelkeypoints[1])):
        cor_x, cor_y = int(labelkeypoints[1][n][0]), int(labelkeypoints[1][n][1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), 2)
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(img, start_xy, end_xy, (0, 255, 0), 1)
    return img

def main(ratio, video_name, outputinfo):
    video = skvideo.io.vread(video_name)
    data = loadCsv(outputinfo)

    print('Load/Write BBox')
    for frame in range(1831):
        all_bboxinfo_inoneframe = []
        print(frame)
        for eachdata in data:
            frameid = json.loads(eachdata[11])
            if frameid <= frame and frameid > (frame - ratio):
                label = json.loads(eachdata[5])
                bbox = json.loads(eachdata[3])
                labelbbox = [label,bbox[0],bbox[1],bbox[2],bbox[3]]
                all_bboxinfo_inoneframe.append(labelbbox)
        if frame > video.shape[0]:
            continue
        tmp = video[frame, ...]
        for i in range(len(all_bboxinfo_inoneframe)):
            tmp = draw_bbox(tmp, all_bboxinfo_inoneframe[i])
        video[frame, ...] = tmp

    print('Load/Write Keypoints')
    for frame in range(1831):
        all_keypointsinfo_inoneframe = []
        print(frame)
        for eachdata in data:
            frameid = json.loads(eachdata[11])
            if frameid <= frame and frameid > (frame - ratio):
                label = json.loads(eachdata[5])
                keypoints = json.loads(eachdata[10])
                labelkeypoints = [label, keypoints]
                all_keypointsinfo_inoneframe.append(labelkeypoints)
        if frame > video.shape[0]:
            continue
        tmp = video[frame, ...]
        for i in range(len(all_keypointsinfo_inoneframe)):
            if all_keypointsinfo_inoneframe[i][0] == 1:
                tmp = draw_car(tmp, all_keypointsinfo_inoneframe[i])
            else:
                tmp = draw_people(tmp, all_keypointsinfo_inoneframe[i])
        video[frame, ...] = tmp

    skvideo.io.vwrite(video_name[:-4] + "_bbox.mp4", video)


if __name__ == "__main__":
    main(ratio=3, video_name='../input/video.mp4',outputinfo='../output_new/output.csv')
