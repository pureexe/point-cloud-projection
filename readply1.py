from plyfile import PlyData, PlyElement
import os
import numpy as np
import matplotlib.pyplot as plt
import json

ref = 'img434'
dataset = 'notredame_resize_new2'
def findCameraPly(dataset):
  return "meshroom/cloud_and_poses.ply"

def findCameraSfm(dataset):
  return "meshroom/cameras.sfm"

def findCameraFocal(dataset):
  return "meshroom/cameras_focal.txt"

path_sfm =  findCameraSfm(dataset)
path_focal = findCameraFocal(dataset)
path_ply = findCameraPly(dataset)

with open(path_sfm, "r") as f:
   js = json.load(f)
   for pose in js['poses']:
       if pose['poseId'] == '1755348266':
           r = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
           t = -r*np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
           cons = np.matrix([0,0,0,1])
           extrinsic = np.concatenate((r,t), axis = 1)
           extrinsic = np.concatenate((extrinsic, cons), axis = 0)
           print(extrinsic)
           break

with open(path_focal, "r") as f:
   js = json.load(f)
   for i in range (len(js)):
       for j in range(len(js[str(i)]['poseId'])):
           if js[str(i)]['poseId'][j] == '1755348266':
               p0 = float(js[str(i)]['principalPoint'][0])
               p1 = float(js[str(i)]['principalPoint'][1])
               fx =  float(js[str(i)]['pxFocalLength'])
               fy = float(js[str(i)]['pyFocalLength'])
               intrinsic = np.matrix([[fx,0,p0], [0,fy,p1], [0,0,1]])
               print(intrinsic)
               break

with open(path_ply, 'rb') as f:
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


color1 =color1/255

print("==== INTRINSIC ====")
print(intrinsic)
print("==== EXTRINSIC ====")
print(extrinsic)
print("==== coordinate ====")
print(coordinate1[:,0])
exit()
coordinate1 = np.dot(intrinsic, np.dot(extrinsic, coordinate1)[0:3,:])


for i in range (coordinate1.shape[1]):
    z= coordinate1[2, i]
    coordinate1[:,i] = coordinate1[:, i]/z

u = coordinate1[0,:]
v = coordinate1[1,:]

count = 0
should_be_color = np.zeros((1200,900,3))
for i in range(15477):
    for j in range(4):
        for k in range (4):
            should_be_color[int(v[0,i])+j:int(v[0,i])+j+1, int(u[0,i])+k:int(u[0,i])+k+1,0] = color1[0, i]
            should_be_color[int(v[0,i])+j:int(v[0,i])+j+1, int(u[0,i])+k:int(u[0,i])+k+1,1] = color1[1, i]
            should_be_color[int(v[0,i])+j:int(v[0,i])+j+1, int(u[0,i])+k:int(u[0,i])+k+1,2] = color1[2, i]
    count += 1
plt.imshow(should_be_color)
plt.show()
