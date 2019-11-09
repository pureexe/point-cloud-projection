from plyfile import PlyData, PlyElement
import os
import numpy as np
import matplotlib.pyplot as plt
import json

ref = 642
dataset = 'notredame_resize_new2'
dataset = 'notredame_resize_new2'
def findCameraPly(dataset):
  return "meshroom/cloud_and_poses.ply"

def findCameraSfm(dataset):
  return "meshroom/cameras.sfm"

def findCameraFocal(dataset):
  return "meshroom/cameras_focal.txt"

def intrinsic(path):
    with open(path, "r") as f:
        js = json.load(f)
        for pose in js['poses']:
            if pose['poseId'] == '1755348266':
                r = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
                t = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
                cons = np.matrix([0,0,0,1])
                intrinsic_val = np.concatenate((r,t), axis = 1)
                print(intrinsic_val)
                intrinsic_val = np.concatenate((intrinsic_val, cons), axis = 0)
    print(intrinsic_val)

    return intrinsic_val

def extrinsic(path):
    with open(path, "r") as f:
        js = json.load(f)
        for i in range (len(js)):
            for j in range(len(js[str(i)]['poseId'])):
                if js[str(i)]['poseId'][j] == '1755348266':
                    p0 = float(js[str(i)]['principalPoint'][0])
                    p1 = float(js[str(i)]['principalPoint'][1])
                    #p0 = 768
                    #p1 = 1152
                    fx =  float(js[str(i)]['pxFocalLength'])
                    fy = float(js[str(i)]['pyFocalLength'])
                    #fx = 2319.6343126489392
                    #fy = 2319.6343126489392
                    extrinsic = np.matrix([[fx,0,p0], [0,fy,p1], [0,0,1]])
    return extrinsic

def ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        data = plydata.elements[0].data
        data = np.reshape(data, [15477, 1])
        coordinate1 = np.matrix([[plydata.elements[0].data[0][0]],\
        [plydata.elements[0].data[0][1]], \
        [plydata.elements[0].data[0][2]],[1]])
        color1 = np.matrix([[plydata.elements[0].data[0][3]],\
        [plydata.elements[0].data[0][4]],\
        [plydata.elements[0].data[0][5]]])
        for j in range(1, data.shape[0]):
            coordinate2 = np.matrix([[plydata.elements[0].data[j][0]],[plydata.elements[0].data[j][1]], [plydata.elements[0].data[j][2]],[1]])
            color2 = np.matrix([[plydata.elements[0].data[j][3]],[plydata.elements[0].data[j][4]], [plydata.elements[0].data[j][5]]])
            coordinate1 = np.concatenate((coordinate1, coordinate2), axis =1)
            color1 = np.concatenate((color1, color2), axis = 1)
    return (coordinate1, color1)

path_sfm =  findCameraSfm(dataset)
path_focal = findCameraFocal(dataset)
path_ply = findCameraPly(dataset)

intrinsic = intrinsic(path_sfm)
extrinsic = extrinsic(path_focal)
(coordinate, color) = ply(path_ply)
print(intrinsic)
print(extrinsic)
color =color/255
coordinate = np.dot(extrinsic, np.dot(intrinsic, coordinate)[0:3,:])

for i in range (coordinate.shape[1]):
    z= coordinate[2, i]
    a = coordinate[:, i]/z
    coordinate[:,i] = a
u = coordinate[0,:]
v = coordinate[1,:]
u_min = np.min(u)
u_max = np.max(u)
v_min = np.min
#u = (coordinate[0,:]+7685)/9653*1200
#v = (coordinate[1,:]+131)/1308*900
#u = (coordinate[0,:]+13831)/17341*1200
#v = (coordinate[1,:]+167)/2348*900
#u = (coordinate1[0,:]+33414)/285208*1200
#v = (coordinate1[1,:]+7444)/65476*900
print(np.min(u))
print(np.max(u))
print(np.min(v))
print(np.max(v))

should_be_color = np.zeros((1200,900,3))
for i in range(15477):
    #should_be_color[int(u[0,i]), int(v[0,i]),:] = color1[:, i]
    should_be_color[int(u[0,i]), int(v[0,i]),0] = color[0, i]
    should_be_color[int(u[0,i]), int(v[0,i]),1] = color[1, i]
    should_be_color[int(u[0,i]), int(v[0,i]),2] = color[2, i]

plt.imshow(should_be_color)
plt.show()
