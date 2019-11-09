from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

focal = 30
mat = loadmat('data/0001_mesh_rightfar.mat')
color = mat['colors']
vertex = mat['vertices']
camera_matrix = np.loadtxt('data/0001_camera_matrix_rightfar.txt')

# Extensic
extrinsic = camera_matrix

#Intrinsic
intrinsic = np.array([
    [focal, 0, 128],
    [0, focal, 128],
    [0, 0, 1]
])

#Add 1 into vextex last channel
position = np.ones([vertex.shape[0], 4])
position[:,:3] = vertex

# APPLY CAMERA MODEL
#projected = np.matmul(intrinsic,np.matmul(extrinsic,position.transpose()))
projected = np.matmul(intrinsic,position.transpose()[:3])


#NORMALLIZE
projected = projected / projected[2,:]
projected = projected.transpose()
projected[:,0] = np.round(projected[:,0])
projected[:,1] = np.round(projected[:,1])
projected = projected.astype(np.int32)
image = np.zeros((256,256,3))
for i in range(len(color)):
    try:
        u,v,_ = projected[i]
        image[v,u,:] = color[i]
    except:
        pass
plt.imsave("50_50.png",image)
plt.imshow(image)
plt.show()

