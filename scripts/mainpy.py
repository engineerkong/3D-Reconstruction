from s1_planefitting import planefitting
from s2_lidar3dto2d import lidar3dto2d
from s3_matching2d import matching2d
from s4_get3dlidar import get3dlidar
from s5_removeoutlier1 import removeoutlier1
from s6_findnearpoints import findnearpoints
from s7_removeoutlier2 import removeoutlier2
from s8_icp3d import icp3d_obj
from s9_findorientation import findorientation
from s10_findskelett import findskelett

planefitting(path_in_lidar='../input/lidarcloud/',path_out='../result_new/res1/')
lidar3dto2d(path_in='../result_new/res1/',path_out='../result_new/res2/')
matching2d(path_in_people='../input/peoplekeypoints.json',path_in_carkeypoints='../input/carkeypoints/',
           path_in_lidar2d='../result_new/res2/',ratio=3,
           path_out_people='../result_new/res3people/',path_out_car='../result_new/res3car/')
get3dlidar(path_in_car='../result_new/res3car/',path_in_people='../result_new/res3people/',
           path_in_lidar='../result_new/res1/',path_out_car='../result_new/res4car/',path_out_people='../result_new/res4people/')
removeoutlier1(path_in_car='../result_new/res4car/',path_in_people='../result_new/res4people/',
              path_out_car='../result_new/res5car/',path_out_people='../result_new/res5people/')
findnearpoints(path_in_car='../result_new/res5car/',path_in_people='../result_new/res5people/',
               path_in_lidar='../result_new/res1/',path_out_car='../result_new/res6car/',path_out_people='../result_new/res6people/')
removeoutlier2(path_in_car='../result_new/res6car/',path_in_people='../result_new/res6people/',
              path_out_car='../result_new/res7car/',path_out_people='../result_new/res7people/')
icp3d_obj(path_in_carkeypoints='../result_new/res7car/',path_in_model_obj='../input/carmesh/',
          path_out_t='../output_new/car_translation/',path_out_a='../output_new/car_yaw/')
findorientation(path_in_peoplekeypoints='../result_new/res7people/',path_out_people='../output_new/people_lwhyaw/')
findskelett(path_in_people2d='../input/peoplekeypoints.json',path_in_people3d='../result_new/res7people/',ratio=3,path_out_people='../output_new/people_skelett/')